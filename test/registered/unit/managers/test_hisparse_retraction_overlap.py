"""Retraction must preserve the output prefix, not an overlapped pending step."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestHiSparseRetractionOverlap(CustomTestCase):
    def make_fixture(self, prefix_len, ahead):
        frontier = prefix_len + ahead
        coordinator = object.__new__(HiSparseCoordinator)
        coordinator.spec_cache = None
        coordinator.device = "cpu"
        coordinator.device_buffer_size = 4
        coordinator.compress_ratio = 1
        coordinator._skip_first_backup = [False] * 3
        coordinator._has_pending_backup = False
        coordinator._backup_done_event = MagicMock()
        coordinator.decode_backup_stream = MagicMock()
        coordinator.decode_producer_stream = MagicMock()
        coordinator.draft_pool = None
        coordinator.admit_request_direct = MagicMock()
        coordinator.req_to_host_pool = torch.stack(
            [torch.arange(128), torch.arange(128) + 64, torch.arange(128) + 256]
        )
        coordinator.req_to_host_pool_allocated_len = torch.tensor([0, frontier, 0])
        coordinator.req_to_device_buffer = torch.stack(
            [torch.arange(68) + 100 * row for row in range(3)]
        )
        coordinator.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.stack(
                [torch.arange(128) + 128 * row for row in range(3)]
            )
        )

        host = torch.full((2, 512, 1, 1), -999, dtype=torch.int64)
        device = torch.full_like(host, -888)
        index_k = torch.full((512, 2), -777, dtype=torch.int64)
        expected = torch.arange(prefix_len)[None, :, None, None].repeat(2, 1, 1, 1)
        expected[1] += 1000
        host[:, 64 : 64 + prefix_len] = expected
        for position in range(min(frontier, coordinator.device_buffer_size)):
            device[:, 100 + position, 0, 0] = torch.tensor([position, position + 1000])
        # Only the newest device token has not yet reached host. In the overlap
        # case it is outside the prefix to resume, and shares the reserved slot
        # with the previous token: flushing it at prefix_len-1 would corrupt KV.
        host[:, 64 + frontier - 1] = -999
        slot = min(frontier - 1, coordinator.device_buffer_size)
        latest_loc = coordinator.req_to_device_buffer[1, slot]
        device[:, latest_loc, 0, 0] = torch.tensor([frontier - 1, frontier + 999])
        index_k[128 : 128 + frontier] = torch.arange(frontier)[:, None]

        allocations = []

        def allocate_host(mapping, allocated_lens, row, start, count):
            allocations.append((start, count))
            return mapping[row, start : start + count]

        def backup_from_device(pool, host_locs, device_locs, **kwargs):
            host[:, host_locs] = pool.kv_buffer[:, device_locs]

        coordinator.mem_pool_host = SimpleNamespace(
            kv_buffer=host,
            alloc_paged_token_slots=allocate_host,
            backup_from_device_all_layer=backup_from_device,
        )
        coordinator.mem_pool_device = SimpleNamespace(
            kv_buffer=device,
            index_key_cache=SimpleNamespace(
                cpu_copy=lambda indices: index_k[indices].clone(),
                load_cpu_copy=lambda data, indices: index_k.index_copy_(
                    0, indices, data
                ),
            ),
        )
        req = SimpleNamespace(
            rid="non-spec-overlap",
            origin_input_ids=list(range(prefix_len - 1)),
            output_ids=[123, 456],
            kv=SimpleNamespace(
                req_pool_idx=1,
                kv_committed_len=frontier,
                kv_allocated_len=frontier,
                retraction_backup=None,
            ),
        )
        return coordinator, req, expected, allocations, index_k

    def test_non_spec_overlap_and_page_boundaries_restore_the_output_prefix(self):
        for prefix_len in (3, 63, 64, 65):
            for ahead in (0, 1):
                with self.subTest(prefix_len=prefix_len, ahead=ahead):
                    co, req, expected, _, index_k = self.make_fixture(prefix_len, ahead)
                    with patch(
                        "sglang.srt.managers.hisparse_coordinator.device_module"
                    ):
                        co.backup_for_retraction(req)
                    backup = req.kv.retraction_backup.cpu_tensors
                    self.assertEqual(backup.num_tokens, prefix_len)
                    for layer in range(2):
                        torch.testing.assert_close(
                            backup.host_kv[layer], expected[layer]
                        )

                    # The pending output is discarded on retract; output_ids
                    # remains unchanged. The real preallocator reconstructs the
                    # prefix length from those IDs, not from the old KV cursor.
                    req.kv.kv_committed_len = DecodePreallocQueue._pre_alloc_fill_len(
                        req
                    )
                    req.kv.req_pool_idx = 2
                    co.mem_pool_host.kv_buffer.fill_(-555)
                    index_k.fill_(-444)
                    co.restore_after_retraction(req)
                    torch.testing.assert_close(
                        co.mem_pool_host.kv_buffer[:, 256 : 256 + prefix_len], expected
                    )
                    torch.testing.assert_close(
                        index_k[256 : 256 + prefix_len],
                        torch.arange(prefix_len)[:, None].expand(-1, 2),
                    )
                    co.admit_request_direct.assert_called_once_with(
                        req, load_draft=False
                    )
                    self.assertIsNone(req.kv.retraction_backup)

    def test_uncomputed_output_prefix_is_rejected_before_backup(self):
        co, req, _, allocations, _ = self.make_fixture(64, -1)
        with (
            patch("sglang.srt.managers.hisparse_coordinator.device_module"),
            self.assertRaisesRegex(ValueError, "exceeds available KV"),
        ):
            co.backup_for_retraction(req)
        self.assertEqual(allocations, [])
        self.assertIsNone(req.kv.retraction_backup)


if __name__ == "__main__":
    unittest.main()

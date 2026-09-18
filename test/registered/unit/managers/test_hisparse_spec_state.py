import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.managers.hisparse_spec_state import (
    HiSparseSpecCache,
    make_hisparse_spec_layout,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestHiSparseSpecState(CustomTestCase):
    def test_verify_token_padding_is_not_counted_as_another_draft_step(self):
        coordinator = object.__new__(HiSparseCoordinator)
        coordinator.spec_num_draft_tokens = 3
        coordinator.top_k = 2048
        coordinator.attention_buffer_size = 8256
        coordinator.spec_cache = SimpleNamespace(states={0: object()})
        coordinator.req_device_buffer_tokens = torch.zeros(
            (1, 2, 8320), dtype=torch.int32
        )
        coordinator.req_device_buffer_token_locs = (
            coordinator.req_device_buffer_tokens.clone()
        )
        coordinator.req_to_host_pool = torch.zeros((2, 128), dtype=torch.int64)
        coordinator.mem_pool_host = SimpleNamespace(kv_buffer=[torch.zeros(1)])
        coordinator.mem_pool_device = SimpleNamespace(kv_buffer=[torch.zeros(1)])
        coordinator.top_k_device_locs_buffer = torch.zeros((2, 6144), dtype=torch.int32)
        coordinator.num_real_reqs = torch.tensor([1], dtype=torch.int32)
        tokens = torch.arange(4 * 2048, dtype=torch.int32).view(4, 2048)
        with patch(
            "sglang.srt.managers.hisparse_coordinator.load_cache_to_device_buffer_spec_mla"
        ) as kernel:
            output = coordinator._run_swap_in_kernel(
                torch.tensor([1]), torch.tensor([101, 102, 103]), tokens, 0
            )
        self.assertEqual(output.shape, (3, 2048))
        rows = kernel.call_args.kwargs["top_k_tokens"]
        self.assertEqual(rows.shape, (1, 3, 2048))
        torch.testing.assert_close(rows[0], tokens[:3])

    def test_draft_receive_uses_host_sources_and_logical_gpu_destinations(self):
        coordinator = object.__new__(HiSparseCoordinator)
        coordinator.spec_cache = object()
        coordinator.is_dsv4_hisparse = False
        coordinator.alloc_device_buffer = MagicMock()
        coordinator._host_committed_lens = [0, 0]
        coordinator._skip_first_backup = [False, False]
        coordinator.req_device_buffer_tokens = torch.zeros((2, 2, 4), dtype=torch.int32)
        coordinator.req_to_host_pool = torch.tensor([[0, 0, 0], [5, 8, 9]])
        coordinator.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.tensor([[0, 0, 0], [64, 65, 66]], dtype=torch.int32)
        )
        coordinator.draft_pool = SimpleNamespace(layer_num=1)
        coordinator.draft_host_pool = SimpleNamespace(
            load_to_device_per_layer=MagicMock()
        )
        req = SimpleNamespace(
            rid="draft-transfer-domain",
            kv=SimpleNamespace(req_pool_idx=1, kv_allocated_len=3, kv_committed_len=3),
        )
        coordinator.admit_request_direct(req)
        args = coordinator.draft_host_pool.load_to_device_per_layer.call_args.args
        torch.testing.assert_close(args[1], torch.tensor([5, 8, 9]))
        torch.testing.assert_close(args[2], torch.tensor([64, 65, 66]))
        self.assertEqual(args[3], 0)
        self.assertFalse(req.hisparse_staging)
        self.assertEqual(coordinator._host_committed_lens[1], 3)

    def test_lookahead_mapping_covers_page_wrap_for_multiple_requests(self):
        coordinator = object.__new__(HiSparseCoordinator)
        coordinator.spec_cache = object()
        coordinator.page_size = 64
        coordinator.device_buffer_size = 8192
        coordinator.device = "cpu"
        coordinator._backup_speculative_committed = MagicMock()
        coordinator.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.stack(
                [torch.arange(256) + row * 256 for row in range(3)]
            )
        )
        coordinator.req_to_device_buffer = torch.stack(
            [torch.arange(8320) + row * 10000 for row in range(3)]
        )
        coordinator.mem_pool_device = SimpleNamespace(
            full_to_hisparse_device_index_mapping=torch.zeros(768, dtype=torch.int64)
        )
        coordinator.req_device_buffer_tokens = torch.full(
            (2, 3, 8320), -1, dtype=torch.int32
        )
        coordinator.req_device_buffer_token_locs = torch.full(
            (2, 3, 8320), -1, dtype=torch.int32
        )
        reqs = [
            SimpleNamespace(kv=SimpleNamespace(kv_committed_len=n)) for n in [63, 129]
        ]
        coordinator.prepare_speculative_decode(
            reqs=reqs, req_pool_indices=torch.tensor([1, 2]), reserve=8
        )
        mapping = coordinator.mem_pool_device.full_to_hisparse_device_index_mapping
        self.assertEqual(mapping[256 + 63].item(), 18255)
        self.assertEqual(mapping[256 + 64].item(), 18192)
        self.assertEqual(mapping[256 + 70].item(), 18198)
        self.assertEqual(mapping[512 + 129].item(), 28193)
        self.assertEqual(mapping[512 + 136].item(), 28200)
        self.assertEqual(coordinator.req_device_buffer_tokens[1, 1, 8192].item(), 64)
        self.assertEqual(
            coordinator.req_device_buffer_token_locs[0, 2, 8200].item(), 28200
        )
        coordinator._backup_speculative_committed.assert_called_once_with(reqs=reqs)

    def test_scratch_covers_union_capacity_after_page_rounding(self):
        for hot, steps, expected in [(4096, 4, 4096), (4096, 3, 2048), (8192, 4, 64)]:
            with self.subTest(hot=hot, steps=steps):
                layout = make_hisparse_spec_layout(
                    num_draft_tokens=steps,
                    top_k=2048,
                    hot_size=hot,
                    page_size=64,
                    req_slots=3,
                    shared_layers=(False, True),
                )
                self.assertEqual(layout.scratch_size, expected)
                self.assertGreaterEqual(hot + layout.scratch_size, steps * 2048)
        with self.assertRaisesRegex(ValueError, ">= 4096"):
            make_hisparse_spec_layout(
                num_draft_tokens=4,
                top_k=2048,
                hot_size=4096,
                page_size=64,
                req_slots=3,
                shared_layers=(False,),
                scratch_size=64,
            )

    def test_standalone_anchors_and_packed_metadata_alignment(self):
        layout = make_hisparse_spec_layout(
            num_draft_tokens=3,
            top_k=1025,
            hot_size=4096,
            page_size=64,
            req_slots=3,
            shared_layers=(False, False, False, True, True, True),
        )
        self.assertEqual(layout.anchors, (0, 1, 2))
        self.assertEqual(layout.layer_anchors, (0, 1, 2, 2, 2, 2))
        self.assertGreaterEqual(layout.metadata_width, 5 * 3 * 1025)
        self.assertEqual(layout.metadata_width % 2, 0)

    def test_slot_reuse_resets_state_without_aliasing_anchors_or_other_requests(self):
        layout = make_hisparse_spec_layout(
            num_draft_tokens=4,
            top_k=2048,
            hot_size=8192,
            page_size=64,
            req_slots=3,
            shared_layers=(False, True, False, True),
        )
        cache = HiSparseSpecCache(layout=layout, device="cpu")
        locs = torch.arange(layout.scratch_size, dtype=torch.int32)
        cache.reset_request(req_index=1, scratch_locs=locs)
        first, second = cache.states[0], cache.states[2]
        first.scratch_locs[1, 0] = 777
        self.assertEqual(second.scratch_locs[1, 0].item(), 0)
        self.assertEqual(locs[0].item(), 0)
        for state in cache.states.values():
            state.cache_index[1].fill_(123)
            state.cache_index[2].fill_(456)
            state.cache_policy[0, 1] = 7
            state.cache_policy[2].fill_(8)
            state.scratch_state[2].fill_(9)
            for bank in range(4):
                state.scratch_state[0, bank * layout.req_slots + 1] = 10
                state.scratch_state[0, bank * layout.req_slots + 2] = 11
        cache.reset_request(req_index=1)
        for state in cache.states.values():
            self.assertTrue(state.cache_index[1].eq(-1).all())
            self.assertTrue(state.cache_index[2].eq(456).all())
            self.assertEqual(state.cache_policy[0, 1].item(), 0)
            self.assertTrue(state.cache_policy[2].eq(0).all())
            self.assertTrue(state.scratch_state[2].eq(-1).all())
            self.assertTrue(state.scratch_locs[1].eq(-1).all())
            for bank in range(4):
                self.assertEqual(
                    state.scratch_state[0, bank * layout.req_slots + 1].item(), 0
                )
                self.assertEqual(
                    state.scratch_state[0, bank * layout.req_slots + 2].item(), 11
                )


if __name__ == "__main__":
    unittest.main()

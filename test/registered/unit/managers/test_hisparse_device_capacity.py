"""Physical HiSparse growth must be checked before mutating a decode batch."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.allocator.hisparse import HiSparseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestHiSparseDeviceCapacity(CustomTestCase):
    PAGE = 64
    HOT = 8192
    PADDED = HOT + PAGE

    def make_coordinator(self, capacities, free):
        co = object.__new__(HiSparseCoordinator)
        co.spec_cache = None
        co.is_dsv4_hisparse = False
        co.device_buffer_size = self.HOT
        co.padded_buffer_size = self.PADDED
        co.mem_pool_device = SimpleNamespace(page_size=self.PAGE)
        co.req_device_buffer_size = torch.tensor(capacities, dtype=torch.int64)
        co.req_to_device_buffer = torch.zeros(
            (len(capacities), self.PADDED), dtype=torch.int64
        )
        co.req_device_buffer_token_locs = torch.zeros(
            (2, len(capacities), self.PADDED), dtype=torch.int64
        )

        def paged(size):
            pool = PagedTokenToKVPoolAllocator(
                size=size,
                page_size=self.PAGE,
                dtype=torch.bfloat16,
                device="cpu",
                kvcache=MagicMock(),
                need_sort=False,
            )
            pool.evict_to_free_tokens = MagicMock()
            return pool

        allocator = object.__new__(HiSparseTokenToKVPoolAllocator)
        allocator.page_size = self.PAGE
        allocator.speculative_decode = False
        allocator._device_buffer_pages = None
        allocator.logical_attn_allocator = paged(65536)
        allocator.hisparse_attn_allocator = paged(sum(capacities) + free)
        allocator.evict_to_free_tokens = MagicMock()
        co.token_to_kv_pool_allocator = allocator
        for row, cap in enumerate(capacities):
            if cap:
                locs = allocator.hisparse_attn_allocator.alloc(cap)
                co.req_to_device_buffer[row, :cap] = locs
                co.req_device_buffer_token_locs[:, row, :cap] = locs
        return co

    def make_batch(self, co, lengths):
        batch = object.__new__(ScheduleBatch)
        batch.hisparse_coordinator = co
        batch.spec_algorithm = SimpleNamespace(is_none=lambda: True)
        batch.token_to_kv_pool_allocator = co.token_to_kv_pool_allocator
        batch.tree_cache = MagicMock()
        batch.seq_lens_cpu = torch.tensor(lengths, dtype=torch.int64)
        batch.req_pool_indices_cpu = torch.arange(len(lengths), dtype=torch.int64)
        batch.reqs = [
            SimpleNamespace(
                rid=f"req-{row}",
                beam_group=None,
                kv=SimpleNamespace(kv_committed_len=n, req_pool_idx=row),
            )
            for row, n in enumerate(lengths)
        ]
        return batch

    def make_queue(self, co, transfers=0):
        queue = object.__new__(DecodePreallocQueue)
        queue.scheduler = SimpleNamespace(enable_hisparse=True, hisparse_coordinator=co)
        queue.token_to_kv_pool_allocator = co.token_to_kv_pool_allocator
        queue.transfer_queue = SimpleNamespace(
            queue=[object() for _ in range(transfers)]
        )
        return queue

    def test_last_short_page_needs_two_physical_pages(self):
        co = self.make_coordinator([8128], free=64)
        batch = self.make_batch(co, [8128])
        self.assertEqual(batch.new_tokens_required_next_decode(), 64)
        self.assertFalse(batch.check_decode_mem())
        # A failed preflight must not change lengths or allocate either pool.
        self.assertEqual(batch.reqs[0].kv.kv_committed_len, 8128)
        self.assertEqual(batch.seq_lens_cpu.tolist(), [8128])
        self.assertEqual(co.req_device_buffer_size.tolist(), [8128])
        self.assertEqual(
            co.token_to_kv_pool_allocator.hisparse_attn_allocator.available_size(), 64
        )

    def test_growth_check_matches_real_page_allocations(self):
        co = self.make_coordinator([64, 8128, self.PADDED], free=192)
        batch = self.make_batch(co, [64, 8128, 8256])
        self.assertTrue(batch.check_decode_mem())
        next_lengths = batch.seq_lens_cpu + 1
        slots = co._grow_device_buffers(
            next_lengths,
            batch.req_pool_indices_cpu,
            next_lengths,
            batch.req_pool_indices_cpu,
        )
        self.assertEqual(
            co.req_device_buffer_size.tolist(), [128, self.PADDED, self.PADDED]
        )
        self.assertEqual(
            co.token_to_kv_pool_allocator.hisparse_attn_allocator.available_size(), 0
        )
        self.assertTrue(torch.all(slots > 0))
        for row, start, end in [(0, 64, 128), (1, 8128, self.PADDED)]:
            torch.testing.assert_close(
                co.req_device_buffer_token_locs[:, row, start:end],
                co.req_to_device_buffer[row, start:end].expand(2, -1),
            )

    def test_selected_rows_exclude_the_retracted_growth(self):
        co = self.make_coordinator([8128, self.PADDED], free=64)
        batch = self.make_batch(co, [8128, 8256])
        self.assertFalse(batch.check_decode_mem(selected_indices=[0]))
        self.assertTrue(batch.check_decode_mem(selected_indices=[1]))
        self.assertTrue(batch.check_decode_mem(selected_indices=[]))

    def test_long_request_can_decode_with_no_free_physical_pages(self):
        co = self.make_coordinator([self.PADDED], free=0)
        batch = self.make_batch(co, [8256])
        self.assertEqual(batch.new_tokens_required_next_decode(), 64)
        self.assertTrue(batch.check_decode_mem())
        co.token_to_kv_pool_allocator.logical_attn_allocator.free_pages = torch.empty(
            0, dtype=torch.int64
        )
        self.assertFalse(batch.check_decode_mem())

    def test_partial_buffers_cannot_repeatedly_overcommit_growth(self):
        co = self.make_coordinator([0, 0, 0], free=2 * self.PADDED)
        queue = self.make_queue(co)
        self.assertEqual(queue._hisparse_available_req_slots(), 2)
        for row, remaining in [(0, 1), (1, 0)]:
            co.req_to_device_buffer[row, :64] = (
                co.token_to_kv_pool_allocator.hisparse_attn_allocator.alloc(64)
            )
            co.req_device_buffer_size[row] = 64
            self.assertEqual(queue._hisparse_available_req_slots(), remaining)
        # Releasing a request returns both its physical pages and its promise.
        co.token_to_kv_pool_allocator.free_hisparse_indices(
            co.req_to_device_buffer[0, :64]
        )
        co.req_device_buffer_size[0] = 0
        self.assertEqual(queue._hisparse_available_req_slots(), 1)

    def test_transfers_reserve_space_after_short_request_growth(self):
        co = self.make_coordinator([64, 64, 0], free=4 * self.PADDED - 128)
        queue = self.make_queue(co, transfers=1)
        self.assertEqual(queue._hisparse_available_req_slots(), 1)

    def test_resumption_waits_for_growth_reservations(self):
        co = self.make_coordinator([64, 64, 0], free=2 * self.PADDED - 128)
        queue = self.make_queue(co)
        queue.retracted_queue = [SimpleNamespace(rid="waiting")]
        queue.req_to_token_pool = SimpleNamespace(available_size=lambda: 10)
        queue._uses_swa_tail_prealloc = lambda: False
        queue._allocatable_token_budgets = lambda **kwargs: 65536
        queue._prealloc_required_tokens = MagicMock(return_value=(64, 0))
        queue._pre_alloc = MagicMock()
        co.restore_after_retraction = MagicMock()
        self.assertEqual(queue.resume_retracted_reqs(), [])
        queue._pre_alloc.assert_not_called()
        co.restore_after_retraction.assert_not_called()
        self.assertEqual(len(queue.retracted_queue), 1)

    def install_retraction_fixture(self, batch):
        co = batch.hisparse_coordinator
        batch._get_decode_retraction_order = lambda reqs: list(
            reversed(range(len(reqs)))
        )

        def release(idx, remaining, offload_kv=True):
            row = batch.reqs[idx].kv.req_pool_idx
            cap = int(co.req_device_buffer_size[row])
            co.token_to_kv_pool_allocator.free_hisparse_indices(
                co.req_to_device_buffer[row, :cap]
            )
            co.req_device_buffer_size[row] = 0
            return True

        def filter_batch(*, keep_indices):
            batch.reqs = [batch.reqs[i] for i in keep_indices]

        batch.release_req = MagicMock(side_effect=release)
        batch.filter_batch = filter_batch

    def test_physical_shortage_uses_existing_retraction_loop(self):
        co = self.make_coordinator([8128, self.PADDED], free=64)
        batch = self.make_batch(co, [8128, 8257])
        self.assertFalse(batch.check_decode_mem())
        victim, survivor = batch.reqs
        self.install_retraction_fixture(batch)
        with patch(
            "sglang.srt.managers.schedule_batch.NewTokenRatioTracker.estimate_new_token_ratio_after_retract",
            return_value=0.5,
        ):
            retracted, _, aborted = batch.retract_decode()
        self.assertEqual(retracted, [victim])
        self.assertEqual(aborted, [])
        self.assertEqual(batch.reqs, [survivor])

    def test_unfit_single_request_is_aborted_without_killing_scheduler(self):
        co = self.make_coordinator([8128], free=64)
        batch = self.make_batch(co, [8128])
        self.assertFalse(batch.check_decode_mem())
        victim = batch.reqs[0]
        self.install_retraction_fixture(batch)
        with patch(
            "sglang.srt.managers.schedule_batch.NewTokenRatioTracker.estimate_new_token_ratio_after_retract",
            return_value=0.5,
        ):
            retracted, _, aborted = batch.retract_decode()
        self.assertEqual(retracted, [])
        self.assertEqual(aborted, [victim])
        self.assertEqual(victim.to_finish.to_json()["type"], "abort")
        self.assertEqual(batch.reqs, [])
        self.assertFalse(batch.release_req.call_args.kwargs["offload_kv"])

    def test_other_decode_modes_keep_their_allocator_capacity_checks(self):
        for spec, dsv4, page_size in [
            (True, False, 64),
            (False, True, 64),
            (False, False, 1),
        ]:
            with self.subTest(spec=spec, dsv4=dsv4, page_size=page_size):
                co = self.make_coordinator([self.PADDED], free=0)
                co.is_dsv4_hisparse = dsv4
                co.can_grow_device_buffers = MagicMock(
                    side_effect=AssertionError("unexpected growth check")
                )
                batch = self.make_batch(co, [8256])
                batch.spec_algorithm = SimpleNamespace(is_none=lambda: not spec)
                batch.token_to_kv_pool_allocator.page_size = page_size
                batch.new_tokens_required_next_decode = MagicMock(return_value=64)
                batch.token_to_kv_pool_allocator.check_decode_capacity = MagicMock(
                    return_value=True
                )
                self.assertTrue(batch.check_decode_mem())
                batch.token_to_kv_pool_allocator.check_decode_capacity.assert_called_once_with(
                    num_tokens=64, tree_cache=batch.tree_cache
                )

    def test_page_zero_is_reserved_and_free_filter_does_not_leak(self):
        co = self.make_coordinator([0], free=128)
        allocator = co.token_to_kv_pool_allocator
        slots = allocator.hisparse_attn_allocator.alloc(128)
        self.assertEqual(int(slots.min()), 64)
        allocator.free_hisparse_indices(torch.cat([torch.tensor([0]), slots]))
        self.assertEqual(allocator.hisparse_attn_allocator.available_size(), 128)


if __name__ == "__main__":
    unittest.main()

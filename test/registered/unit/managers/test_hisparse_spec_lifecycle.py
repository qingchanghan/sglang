"""Exercise real KV pools across verify, host backup, retraction and slot reuse."""

import tempfile
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.allocator.hisparse import HiSparseTokenToKVPoolAllocator
from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, ReqToTokenPool
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


class TestHiSparseSpecLifecycle(CustomTestCase):
    def setUp(self):
        if not torch.cuda.is_available() or torch.version.hip:
            self.skipTest("CUDA is required for speculative HiSparse.")
        self.rendezvous = tempfile.TemporaryDirectory()
        self.addCleanup(self.rendezvous.cleanup)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="gloo",
                init_method=f"file://{self.rendezvous.name}/rendezvous",
                rank=0,
                world_size=1,
            )
            self.addCleanup(torch.distributed.destroy_process_group)
        size, ratio = 24960, 4
        pool_args = dict(
            page_size=64,
            kv_lora_rank=512,
            dtype=torch.bfloat16,
            qk_rope_head_dim=64,
            device="cuda",
            index_head_dim=128,
            enable_memory_saver=False,
            kv_cache_dim=576,
        )
        self.target = HiSparseDSATokenToKVPool(
            size=size, layer_num=2, host_to_device_ratio=ratio, **pool_args
        )
        self.allocator = HiSparseTokenToKVPoolAllocator(
            size=size,
            page_size=64,
            dtype=torch.bfloat16,
            device="cuda",
            kvcache=self.target,
            need_sort=False,
            host_to_device_ratio=ratio,
            speculative_decode=True,
        )
        self.reqs = ReqToTokenPool(
            size=4, max_context_len=32768, device="cuda", enable_memory_saver=False
        )
        self.coordinator = HiSparseCoordinator(
            req_to_token_pool=self.reqs,
            token_to_kv_pool_allocator=self.allocator,
            top_k=2048,
            device_buffer_size=8192,
            device="cuda",
            tp_group=torch.distributed.group.WORLD,
            host_to_device_ratio=ratio,
            shared_index_layers=[False, True],
            speculative_num_draft_tokens=4,
            speculative_scratch_size=64,
        )
        self.addCleanup(self.coordinator.destroy)
        self.draft = DSATokenToKVPool(size=size * ratio, layer_num=1, **pool_args)
        self.coordinator.bind_resident_draft_pool(self.draft)

    @staticmethod
    def _pattern(positions, layer):
        return ((positions + layer * 997) % 30000).to(torch.int16)[..., None, None]

    def _reserve(self, req, end):
        start = req.kv.kv_allocated_len
        if end <= start:
            return
        row = req.kv.req_pool_idx
        last = (
            self.reqs.req_to_token[row, start - 1 : start].to(torch.int64)
            if start
            else torch.tensor([-1], device="cuda", dtype=torch.int64)
        )
        locs = self.allocator.alloc_logical_only(
            prefix_lens=torch.tensor([start], device="cuda"),
            prefix_lens_cpu=torch.tensor([start]),
            seq_lens=torch.tensor([end], device="cuda"),
            seq_lens_cpu=torch.tensor([end]),
            last_loc=last,
            extend_num_tokens=end - start,
        )
        self.assertIsNotNone(locs)
        self.reqs.write((row, slice(start, end)), locs)
        req.kv.kv_allocated_len = end

    def _admit(self, length):
        req = SimpleNamespace(rid="lifecycle", kv=ReqKvInfo(), hisparse_staging=False)
        self.assertIsNotNone(self.reqs.alloc([req]))
        self._reserve(req, length)
        req.kv.kv_committed_len = length
        co = self.coordinator
        host = co.mem_pool_host.alloc_paged_token_slots(
            co.req_to_host_pool,
            co.req_to_host_pool_allocated_len,
            req.kv.req_pool_idx,
            0,
            length,
        ).cpu()
        positions = torch.arange(length)
        for layer in range(2):
            co.mem_pool_host.kv_buffer[layer].view(torch.int16)[host] = self._pattern(
                positions, layer
            )
        co.draft_host_pool.kv_buffer[0].view(torch.int16)[host] = self._pattern(
            positions, 2
        )
        co.admit_request_direct(req)
        logical = self.reqs.req_to_token[req.kv.req_pool_idx, :length].long()
        actual = self.draft.kv_buffer[0].view(torch.int16)[logical].cpu()
        torch.testing.assert_close(
            actual, self._pattern(positions, 2).expand_as(actual)
        )
        return req

    def _verify_step(self, req, accepted):
        start = req.kv.kv_committed_len
        self._reserve(req, start + 8)
        co = self.coordinator
        row = torch.tensor([req.kv.req_pool_idx], device="cuda", dtype=torch.int64)
        co.prepare_speculative_decode(reqs=[req], req_pool_indices=row, reserve=8)
        co.wait_for_pending_backup()
        positions = torch.arange(start, start + 4, device="cuda")
        logical = self.reqs.req_to_token[req.kv.req_pool_idx, positions].long()
        physical = self.allocator.full_to_hisparse_device_index_mapping[logical]
        for layer in range(2):
            self.target.kv_buffer[layer].view(torch.int16)[physical] = self._pattern(
                positions, layer
            )
        self.draft.kv_buffer[0].view(torch.int16)[logical] = self._pattern(positions, 2)
        topk = torch.arange(2048, device="cuda", dtype=torch.int32).repeat(4, 1)
        topk[:, -1] = positions.int()
        seq_lens = (positions + 1).int()
        co.set_num_real_reqs(batch_size=1, unpadded_batch_size=None)
        for layer in range(2):
            locs = co.swap_in_selected_pages(row, seq_lens, topk, layer)
            actual = self.target.kv_buffer[layer].view(torch.int16)[locs.long()]
            torch.testing.assert_close(
                actual, self._pattern(topk.long(), layer).expand_as(actual)
            )
        req.kv.kv_committed_len += accepted

    def _release(self, req):
        row = req.kv.req_pool_idx
        logical = self.reqs.req_to_token[row, : req.kv.kv_allocated_len].clone()
        self.coordinator.request_finished(req)
        self.allocator.free(logical)
        self.reqs.free(req)

    def test_verify_backups_retraction_and_reuse_preserve_kv(self):
        co = self.coordinator
        initial = (
            self.allocator.logical_attn_allocator.available_size(),
            self.allocator.hisparse_attn_allocator.available_size(),
            co.mem_pool_host.available_size(),
        )
        req = self._admit(16384)
        # Variable acceptance crosses a host-page boundary and the extra-page ring.
        for accepted in [1, 2, 4, 3] * 8:
            self._verify_step(req, accepted)
        co.backup_for_retraction(req)
        committed = req.kv.kv_committed_len
        backup = req.kv.retraction_backup
        host = co.req_to_host_pool[req.kv.req_pool_idx, :committed].cpu()
        positions = torch.arange(committed)
        for layer in range(2):
            actual = co.mem_pool_host.kv_buffer[layer].view(torch.int16)[host]
            torch.testing.assert_close(
                actual, self._pattern(positions, layer).expand_as(actual)
            )
        self._release(req)
        # Occupy reused addresses so restore must work with another physical mapping.
        blocker = self._admit(8192)
        restored = self._admit(committed)
        restored.kv.retraction_backup = backup
        co.request_finished(restored)
        # request_finished released host/buffer state, but keep this new logical row.
        restored_host = co.mem_pool_host.alloc_paged_token_slots(
            co.req_to_host_pool,
            co.req_to_host_pool_allocated_len,
            restored.kv.req_pool_idx,
            0,
            committed,
        )
        for layer in range(2):
            co.mem_pool_host.kv_buffer[layer].view(torch.int16)[
                restored_host.cpu()
            ] = -77
        restored_logical = self.reqs.req_to_token[
            restored.kv.req_pool_idx, :committed
        ].long()
        self.draft.kv_buffer[0].view(torch.int16)[restored_logical] = -88
        co.restore_after_retraction(restored)
        actual = self.draft.kv_buffer[0].view(torch.int16)[restored_logical].cpu()
        torch.testing.assert_close(
            actual, self._pattern(positions, 2).expand_as(actual)
        )
        self._verify_step(restored, 3)
        self._release(restored)
        self._release(blocker)
        torch.cuda.synchronize()
        self.assertEqual(
            initial,
            (
                self.allocator.logical_attn_allocator.available_size(),
                self.allocator.hisparse_attn_allocator.available_size(),
                co.mem_pool_host.available_size(),
            ),
        )


if __name__ == "__main__":
    unittest.main()

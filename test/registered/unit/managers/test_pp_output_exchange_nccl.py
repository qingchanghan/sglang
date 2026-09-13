"""Both ends of a PP pair exchanging a four-tensor output dict in one round
(the chunked-prefill hand-off with MTP draft fields) must complete under the
parity order: with every rank sending first the NCCL recvs wait behind the
sends and the ring deadlocks. Real GroupCoordinator send/recv over NCCL on two
GPUs, bounded by a blocking-wait timeout so a regression fails instead of
hanging."""

import os
import unittest

import torch
import torch.multiprocessing as mp
from sglang.srt.managers.scheduler_pp_mixin import pp_output_exchange_send_first
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-c", runner_config="2-gpu-large")

PORT = 29617
WORLD = 2
HIDDEN = 6144


def _output_dict(device: torch.device, seed: int):
    # Shapes and dtypes from the R6 flight recorder: next_token_ids,
    # draft_topk_p, draft_topk_index, draft_hidden_states for bs=1.
    return {
        "next_token_ids": torch.full((1,), seed, dtype=torch.int64, device=device),
        "draft_topk_p": torch.full((1, 1), 0.5, dtype=torch.float32, device=device),
        "draft_topk_index": torch.full((1, 1), seed, dtype=torch.int64, device=device),
        "draft_hidden_states": torch.full(
            (1, HIDDEN), float(seed), dtype=torch.bfloat16, device=device
        ),
    }


def _run(rank: int, world: int, port: int):
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
    torch.cuda.set_device(rank)

    from sglang.srt.distributed.parallel_state import (
        get_pp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="nccl",
        timeout=60,
    )
    initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=world
    )
    pp_group = get_pp_group()
    device = torch.device("cuda", rank)
    peer = (rank + 1) % world
    mine = _output_dict(device, seed=rank + 1)

    if pp_output_exchange_send_first(pp_rank=rank):
        works = pp_group.send_tensor_dict(mine, dst=peer, async_send=True)
        got = pp_group.recv_tensor_dict(src=peer)
    else:
        got = pp_group.recv_tensor_dict(src=peer)
        works = pp_group.send_tensor_dict(mine, dst=peer, async_send=True)
    for p2p in works:
        if p2p.work is not None:
            p2p.work.wait()
    torch.cuda.synchronize(device)

    expected = _output_dict(device, seed=peer + 1)
    assert set(got) == set(expected), got.keys()
    for key, value in expected.items():
        assert torch.equal(got[key], value), key


class TestPPOutputExchangeNccl(CustomTestCase):
    def test_four_tensor_dicts_both_ways_complete(self):
        if torch.cuda.device_count() < WORLD:
            self.skipTest("needs two GPUs")
        mp.spawn(_run, args=(WORLD, PORT), nprocs=WORLD, join=True)


if __name__ == "__main__":
    unittest.main()

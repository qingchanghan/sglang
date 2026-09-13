"""Read-only probe of the target and draft KV pools on a PD decode worker.

For one probed request it fingerprints a few prompt rows of both pools at
fixed lifecycle stages so a missing draft-pool fill shows up as unchanged bytes.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
import torch
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.mem_cache.memory_pool import KVCache

logger = logging.getLogger(__name__)

PROBE_STAGES = ("prealloc", "transferred", "first_batch", "after_gen")
# Generated tokens that must exist before after_gen samples generated rows.
AFTER_GEN_MIN_OUTPUT = 64
AFTER_GEN_SAMPLE = 8

_done_stages: set[tuple[str, str]] = set()
_identity_logged: set[str] = set()


def probe_enabled() -> bool:
    return envs.SGLANG_DRAFT_POOL_PROBE_RID.get() is not None


def _rid_matches(rid: str) -> bool:
    needle = envs.SGLANG_DRAFT_POOL_PROBE_RID.get()
    return needle is not None and needle in rid


def prompt_sample_positions(prompt_len: int) -> list[int]:
    # Head, middle and tail of the prompt, deduplicated and ascending.
    if prompt_len <= 0:
        return []
    mid = prompt_len // 2
    raw = (0, 1, mid, mid + 1, prompt_len - 2, prompt_len - 1)
    return sorted({p for p in raw if 0 <= p < prompt_len})


def index_k_page_offsets(
    *, slot: int, page_size: int, head_dim: int
) -> tuple[int, int, int]:
    # Page row layout (index_buf_accessor): page_size*head_dim K bytes, then
    # page_size fp32 scales; returns (page, k_start, scale_start).
    page, tok = divmod(slot, page_size)
    return page, tok * head_dim, page_size * head_dim + tok * 4


def row_fingerprint(row: torch.Tensor) -> dict:
    flat = row.detach().contiguous().view(-1)
    if flat.dtype != torch.uint8:
        flat = flat.view(torch.uint8)
    data = flat.cpu().numpy()
    return {
        "bytes": int(data.size),
        "nonzero_frac": round(
            float(np.count_nonzero(data)) / max(int(data.size), 1), 4
        ),
        "blake2b8": hashlib.blake2b(data.tobytes(), digest_size=8).hexdigest(),
    }


def sample_layers(pool: KVCache) -> list[int]:
    first = pool.start_layer
    last = first + pool.layer_num - 1
    return sorted({first, first + pool.layer_num // 2, last})


def sample_kv_rows(
    *, pool: KVCache, layer_ids: Iterable[int], slots: list[int]
) -> list[dict]:
    idx = torch.tensor(slots, dtype=torch.long, device=pool.device)
    out = []
    for layer_id in layer_ids:
        rows = pool.get_key_buffer(layer_id)[idx]
        out.append(
            {
                "layer": layer_id,
                "rows": [row_fingerprint(rows[i]) for i in range(len(slots))],
            }
        )
    return out


def sample_index_k_rows(
    *, pool: KVCache, layer_ids: Iterable[int], slots: list[int]
) -> list[dict] | None:
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

    if not isinstance(pool, DSATokenToKVPool):
        return None
    out = []
    for layer_id in layer_ids:
        buf = pool.index_key_cache.get_local_buffer(layer_id)
        # Shared top-k layers keep a 0-row placeholder and never write index-K.
        if buf.shape[0] == 0:
            continue
        rows = []
        for slot in slots:
            page, k_start, _ = index_k_page_offsets(
                slot=slot, page_size=pool.page_size, head_dim=pool.index_head_dim
            )
            rows.append(
                row_fingerprint(buf[page, k_start : k_start + pool.index_head_dim])
            )
        out.append({"layer": layer_id, "rows": rows})
    return out


def pool_identity(pool: KVCache) -> dict:
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

    kv0 = pool.get_key_buffer(pool.start_layer)
    ident = {
        "cls": type(pool).__name__,
        "id": id(pool),
        "layer_num": pool.layer_num,
        "start_layer": pool.start_layer,
        "size": pool.size,
        "page_size": pool.page_size,
        "kv_dtype": str(kv0.dtype),
        "kv0_ptr": kv0.data_ptr(),
        "kv0_shape": list(kv0.shape),
    }
    if isinstance(pool, DSATokenToKVPool):
        ident["index_head_dim"] = pool.index_head_dim
        ident["index_k_pages_per_layer"] = [
            int(b.shape[0]) for b in pool.index_key_cache.buffer
        ]
    return ident


def _kv_ptrs(pool: KVCache) -> set[int]:
    return {
        pool.get_key_buffer(layer_id).data_ptr()
        for layer_id in range(pool.start_layer, pool.start_layer + pool.layer_num)
    }


def _identity(*, scheduler: Scheduler, target: KVCache, draft: KVCache | None) -> dict:
    ident = {
        "target": pool_identity(target),
        "draft": None,
        "same_pool_object": False,
        "draft_kv_ptr_in_target": False,
        "same_allocator": None,
        "same_req_to_token": None,
    }
    draft_worker = scheduler.draft_worker
    if draft_worker is not None:
        ident["same_allocator"] = (
            draft_worker.token_to_kv_pool_allocator
            is scheduler.token_to_kv_pool_allocator
        )
        ident["same_req_to_token"] = (
            draft_worker.req_to_token_pool is scheduler.req_to_token_pool
        )
    if draft is not None:
        ident["draft"] = pool_identity(draft)
        ident["same_pool_object"] = draft is target
        ident["draft_kv_ptr_in_target"] = bool(_kv_ptrs(draft) & _kv_ptrs(target))
    return ident


def _emit(record: dict, *, tp_rank: int) -> None:
    line = json.dumps(record)
    logger.info("[draft-pool-probe] %s", line)
    path = envs.SGLANG_DRAFT_POOL_PROBE_PATH.get()
    if path is None:
        return
    with open(f"{path}.rank{tp_rank}", "a") as f:
        f.write(line + "\n")


def _record(*, scheduler: Scheduler, req: Req, stage: str) -> None:
    target = scheduler.token_to_kv_pool_allocator.get_kvcache()
    draft = (
        scheduler.draft_worker.primary_draft_kv_pool
        if scheduler.draft_worker is not None
        else None
    )
    prompt_len = len(req.origin_input_ids)
    positions = prompt_sample_positions(prompt_len)
    if stage == "after_gen":
        positions = positions + list(range(prompt_len, prompt_len + AFTER_GEN_SAMPLE))
    req_pool_idx = int(req.kv.req_pool_idx)
    slots = scheduler.req_to_token_pool.req_to_token[req_pool_idx, positions].tolist()
    # In-flight forwards write KV on their own stream; drain before reading.
    torch.cuda.synchronize(target.device)
    record = {
        "probe": "draft_pool",
        "stage": stage,
        "rid": req.rid,
        "tp_rank": scheduler.ps.tp_rank,
        "attn_tp_rank": scheduler.ps.attn_tp_rank,
        "dp_rank": scheduler.ps.dp_rank,
        "prompt_len": prompt_len,
        "output_len": len(req.output_ids),
        "req_pool_idx": req_pool_idx,
        "positions": positions,
        "slots": slots,
        "target_kv": sample_kv_rows(
            pool=target, layer_ids=sample_layers(target), slots=slots
        ),
        "target_index_k": sample_index_k_rows(
            pool=target, layer_ids=sample_layers(target), slots=slots
        ),
        "draft_kv": None,
        "draft_index_k": None,
    }
    if draft is not None:
        record["draft_kv"] = sample_kv_rows(
            pool=draft, layer_ids=sample_layers(draft), slots=slots
        )
        record["draft_index_k"] = sample_index_k_rows(
            pool=draft, layer_ids=sample_layers(draft), slots=slots
        )
    if req.rid not in _identity_logged:
        _identity_logged.add(req.rid)
        record["identity"] = _identity(scheduler=scheduler, target=target, draft=draft)
    _emit(record, tp_rank=scheduler.ps.tp_rank)


def maybe_probe_draft_pool(*, scheduler: Scheduler, req: Req, stage: str) -> None:
    if not probe_enabled() or not _rid_matches(req.rid):
        return
    if stage == "after_gen" and len(req.output_ids) < AFTER_GEN_MIN_OUTPUT:
        return
    key = (req.rid, stage)
    if key in _done_stages:
        return
    _done_stages.add(key)
    _record(scheduler=scheduler, req=req, stage=stage)


def maybe_probe_draft_pool_batch(
    *, scheduler: Scheduler, reqs: Iterable[Req], stage: str
) -> None:
    if not probe_enabled():
        return
    for req in reqs:
        maybe_probe_draft_pool(scheduler=scheduler, req=req, stage=stage)

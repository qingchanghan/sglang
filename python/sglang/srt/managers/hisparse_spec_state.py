from __future__ import annotations

from typing import Sequence

import msgspec
import torch

from sglang.kernels.ops.kvcache.hisparse import HiSparseSpecState


class HiSparseSpecLayout(msgspec.Struct, frozen=True, kw_only=True):
    num_draft_tokens: int
    top_k: int
    hot_size: int
    page_size: int
    scratch_size: int
    hash_size: int
    metadata_width: int
    req_slots: int
    anchors: tuple[int, ...]
    layer_anchors: tuple[int, ...]
    buffer_size: int


def make_hisparse_spec_layout(
    *,
    num_draft_tokens: int,
    top_k: int,
    hot_size: int,
    page_size: int,
    req_slots: int,
    shared_layers: Sequence[bool],
    scratch_size: int | None = None,
) -> HiSparseSpecLayout:
    occurrences = num_draft_tokens * top_k
    if not 2 <= num_draft_tokens <= 4 or top_k < 1024 or occurrences > 8192:
        raise ValueError(
            "HiSparse speculative swap requires 2-4 draft tokens, top_k >= 1024, "
            f"and draft_tokens * top_k <= 8192; got {num_draft_tokens=}, {top_k=}."
        )
    if hot_size < top_k or hot_size & (hot_size - 1):
        raise ValueError(
            f"HiSparse hot_size must be a power of two >= {top_k}, got {hot_size}."
        )
    if (
        page_size < 2 * num_draft_tokens
        or not 0 < req_slots <= hot_size
        or not shared_layers
    ):
        raise ValueError(
            "Invalid HiSparse speculative page size or request/layer count."
        )
    minimum_scratch = max(page_size, occurrences - hot_size)
    minimum_scratch = (minimum_scratch + page_size - 1) // page_size * page_size
    scratch_size = minimum_scratch if scratch_size is None else scratch_size
    if (
        not isinstance(scratch_size, int)
        or isinstance(scratch_size, bool)
        or scratch_size < minimum_scratch
        or scratch_size % page_size
    ):
        raise ValueError(
            f"HiSparse scratch_size must be page-aligned and >= {minimum_scratch}; "
            f"got {scratch_size=} and {page_size=}."
        )
    anchors = []
    layer_anchors = []
    for layer, shared in enumerate(shared_layers):
        if not shared:
            anchors.append(layer)
        if not anchors:
            raise ValueError("A shared-index layer must have a preceding anchor.")
        layer_anchors.append(anchors[-1])
    metadata_width = max(4 * req_slots, 5 * occurrences)
    return HiSparseSpecLayout(
        num_draft_tokens=num_draft_tokens,
        top_k=top_k,
        hot_size=hot_size,
        page_size=page_size,
        scratch_size=scratch_size,
        hash_size=1 << (2 * hot_size - 1).bit_length(),
        metadata_width=(metadata_width + 1) // 2 * 2,
        req_slots=req_slots,
        anchors=tuple(anchors),
        layer_anchors=tuple(layer_anchors),
        buffer_size=hot_size + page_size + scratch_size,
    )


class HiSparseSpecCache:
    def __init__(self, *, layout: HiSparseSpecLayout, device: str):
        self.layout = layout
        self.states = {
            layer: self._allocate_state(device=device) for layer in layout.anchors
        }

    def _allocate_state(self, *, device: str) -> HiSparseSpecState:
        layout = self.layout
        scratch_state = torch.full(
            (layout.req_slots + 1, layout.metadata_width),
            -1,
            dtype=torch.int32,
            device=device,
        )
        scratch_state[0].zero_()
        return HiSparseSpecState(
            cache_index=torch.full(
                (layout.req_slots, 2, layout.hash_size),
                -1,
                dtype=torch.int64,
                device=device,
            ),
            cache_policy=torch.zeros(
                (layout.req_slots + 1, layout.hot_size),
                dtype=torch.int32,
                device=device,
            ),
            scratch_locs=torch.full(
                (layout.req_slots, layout.scratch_size),
                -1,
                dtype=torch.int32,
                device=device,
            ),
            scratch_state=scratch_state,
        )

    def reset_request(
        self, *, req_index: int, scratch_locs: torch.Tensor | None = None
    ) -> None:
        for state in self.states.values():
            state.cache_index[req_index].fill_(-1)
            state.cache_policy[0, req_index] = 0
            state.cache_policy[req_index + 1].zero_()
            # The kernel stores four counter banks in row zero, one entry per slot.
            for bank in range(4):
                state.scratch_state[0, bank * self.layout.req_slots + req_index] = 0
            state.scratch_state[req_index + 1].fill_(-1)
            if scratch_locs is None:
                state.scratch_locs[req_index].fill_(-1)
            else:
                state.scratch_locs[req_index].copy_(scratch_locs)

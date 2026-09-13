"""Alignment for offsets applied to padded DSA top-k query rows."""

import torch
import torch.nn.functional as F


def pad_ragged_offsets(offsets: torch.Tensor, num_query_tokens: int) -> torch.Tensor:
    """Keep real-query offsets unchanged; padded top-k rows are all -1.

    Zero is safe for the extra rows because the caller masks those indices.
    Do not truncate excess metadata: that would hide a different alignment bug.
    """
    if offsets.ndim not in (1, 2):
        raise ValueError(f"Expected 1D or 2D ragged offsets, got {offsets.ndim}D")
    extra = num_query_tokens - offsets.shape[0]
    if extra < 0:
        raise ValueError("Ragged offset rows exceed the query rows")
    if extra == 0:
        return offsets
    return F.pad(offsets, (0, extra) if offsets.ndim == 1 else (0, 0, 0, extra))

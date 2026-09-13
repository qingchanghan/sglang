"""Resolve the embedding and lm_head a draft shares with its target.

Under pipeline parallelism the first stage owns embed_tokens and the last stage
owns lm_head; the draft lives on the last stage, so its embedding shard has to
come from the checkpoint instead of the target module.
"""

from __future__ import annotations

import json
import os

import torch
from sglang.srt.layers.utils import PPMissingLayer

EMBED_TOKENS_WEIGHT = "model.embed_tokens.weight"


def target_hosts_embed_tokens(target_model: torch.nn.Module) -> bool:
    return not isinstance(target_model.model.embed_tokens, PPMissingLayer)


def checkpoint_tensor_file(*, model_path: str, weight_name: str) -> str:
    # Sharded checkpoints map each tensor through the safetensors index.
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        return os.path.join(model_path, weight_map[weight_name])
    single_file = os.path.join(model_path, "model.safetensors")
    if os.path.isfile(single_file):
        return single_file
    raise FileNotFoundError(
        f"{model_path}: no model.safetensors.index.json or model.safetensors; "
        f"a pipeline-parallel draft loads {weight_name} from a local checkpoint"
    )


def load_embed_tokens_shard(
    *,
    model_path: str,
    embed_param: torch.nn.Parameter,
    weight_name: str = EMBED_TOKENS_WEIGHT,
) -> None:
    from safetensors import safe_open

    path = checkpoint_tensor_file(model_path=model_path, weight_name=weight_name)
    with safe_open(path, framework="pt", device="cpu") as f:
        full_weight = f.get_tensor(weight_name)
    # The parameter's own loader applies this rank's vocab shard.
    embed_param.weight_loader(embed_param, full_weight)


def resolve_draft_embed_and_head(
    *,
    target_model: torch.nn.Module,
    draft_model: torch.nn.Module,
    model_path: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target_hosts_embed_tokens(target_model):
        return target_model.get_embed_and_head()
    assert not isinstance(target_model.lm_head, PPMissingLayer), (
        "the draft must be hosted on the pipeline stage that owns lm_head"
    )
    # The draft loader skips embed_tokens (shared from the target), so load
    # this stage's shard in place; only the head is borrowed from the target.
    embed_param = draft_model.model.embed_tokens.weight
    load_embed_tokens_shard(model_path=model_path, embed_param=embed_param)
    return embed_param, target_model.lm_head.weight

"""A draft hosted on the last pipeline stage must get its embedding from the
checkpoint (the target there holds only a PPMissingLayer) and borrow only the
lm_head from the target; a target that hosts the embedding keeps sharing both.
Fakes stand in for the models -- CPU only."""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.speculative.draft_embed_tokens import (
    EMBED_TOKENS_WEIGHT,
    checkpoint_tensor_file,
)
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_VOCAB, _HIDDEN = 8, 4


def _write_sharded_checkpoint(model_dir: str, embed: torch.Tensor) -> None:
    shard = "model-00001-of-00002.safetensors"
    save_file({EMBED_TOKENS_WEIGHT: embed}, os.path.join(model_dir, shard))
    with open(os.path.join(model_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": {EMBED_TOKENS_WEIGHT: shard}}, f)


def _embed_param() -> torch.nn.Parameter:
    param = torch.nn.Parameter(torch.zeros(_VOCAB, _HIDDEN))
    # Stands in for VocabParallelEmbedding.weight_loader with a single shard.
    param.weight_loader = lambda p, loaded: p.data.copy_(loaded)
    return param


class _FakeDraftModel:
    def __init__(self):
        self.model = SimpleNamespace(
            embed_tokens=SimpleNamespace(weight=_embed_param())
        )
        self.shared = None

    def set_embed_and_head(self, embed, head):
        self.shared = (embed, head)


def _worker(*, target_model, draft_model, model_path: str) -> EagleDraftWorker:
    # Bypass EagleDraftWorker.__init__ (needs a live draft runner); wire only
    # what init_lm_head reads.
    worker = EagleDraftWorker.__new__(EagleDraftWorker)
    worker.target_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            model=target_model, model_config=SimpleNamespace(model_path=model_path)
        )
    )
    worker.draft_runner = SimpleNamespace(model=draft_model)
    worker.hot_token_id = None
    worker.speculative_algorithm = SpeculativeAlgorithm.EAGLE
    return worker


class TestDraftEmbedTokensUnderPP(CustomTestCase):
    def test_last_stage_target_loads_embed_shard_and_shares_head_only(self):
        expected = torch.arange(_VOCAB * _HIDDEN, dtype=torch.float32).view(
            _VOCAB, _HIDDEN
        )
        head = torch.nn.Parameter(torch.ones(_VOCAB, _HIDDEN))
        target = SimpleNamespace(
            model=SimpleNamespace(embed_tokens=PPMissingLayer()),
            lm_head=SimpleNamespace(weight=head),
        )
        draft = _FakeDraftModel()
        with tempfile.TemporaryDirectory() as model_dir:
            _write_sharded_checkpoint(model_dir, expected)
            _worker(
                target_model=target, draft_model=draft, model_path=model_dir
            ).init_lm_head()

        embed, shared_head = draft.shared
        self.assertIs(embed, draft.model.embed_tokens.weight)
        self.assertTrue(torch.equal(embed.data, expected))
        self.assertIs(shared_head, head)

    def test_target_hosting_embed_still_shares_both(self):
        target_embed = torch.nn.Parameter(torch.full((_VOCAB, _HIDDEN), 2.0))
        head = torch.nn.Parameter(torch.ones(_VOCAB, _HIDDEN))
        target = SimpleNamespace(
            model=SimpleNamespace(embed_tokens=SimpleNamespace(weight=target_embed)),
            lm_head=SimpleNamespace(weight=head),
            get_embed_and_head=lambda: (target_embed, head),
        )
        draft = _FakeDraftModel()
        _worker(
            target_model=target, draft_model=draft, model_path="/nonexistent"
        ).init_lm_head()

        self.assertEqual(draft.shared, (target_embed, head))
        # The draft's own (unloaded) embedding is left alone.
        self.assertTrue(
            torch.equal(
                draft.model.embed_tokens.weight.data, torch.zeros(_VOCAB, _HIDDEN)
            )
        )

    def test_checkpoint_lookup_falls_back_to_single_file_then_fails(self):
        with tempfile.TemporaryDirectory() as model_dir:
            single = os.path.join(model_dir, "model.safetensors")
            save_file({EMBED_TOKENS_WEIGHT: torch.zeros(2, 2)}, single)
            self.assertEqual(
                checkpoint_tensor_file(
                    model_path=model_dir, weight_name=EMBED_TOKENS_WEIGHT
                ),
                single,
            )
        with (
            tempfile.TemporaryDirectory() as empty_dir,
            self.assertRaises(FileNotFoundError),
        ):
            checkpoint_tensor_file(
                model_path=empty_dir, weight_name=EMBED_TOKENS_WEIGHT
            )


if __name__ == "__main__":
    unittest.main()

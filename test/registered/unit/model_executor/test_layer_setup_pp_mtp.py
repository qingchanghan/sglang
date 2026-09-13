"""A PD prefill target may be pipeline-split with EAGLE on (the draft lives on
the last stage and only draft-extend runs there), other roles keep rejecting a
split MTP target under spec, and the draft runner's own 1-layer range is never
affected. Pure config math -- CPU only."""

import unittest
from types import SimpleNamespace

from sglang.srt.model_executor.model_runner_components.layer_setup import (
    resolve_layer_indices,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_NUM_LAYERS = 78


def _mtp_model_config() -> SimpleNamespace:
    return SimpleNamespace(
        num_nextn_predict_layers=1,
        num_hidden_layers=_NUM_LAYERS,
        num_attention_layers=_NUM_LAYERS,
        hf_config=SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"]),
    )


def _resolve(*, model, is_draft_worker: bool):
    return resolve_layer_indices(
        model=model,
        model_config=_mtp_model_config(),
        is_draft_worker=is_draft_worker,
        spec_algorithm=SpeculativeAlgorithm.EAGLE,
    )


class TestLayerSetupPipelineParallelMtp(CustomTestCase):
    def test_prefill_role_accepts_split_mtp_target(self):
        last_stage = SimpleNamespace(start_layer=39, end_layer=_NUM_LAYERS)
        with get_context().override_server_args(
            model_path="dummy", disaggregation_mode="prefill"
        ):
            info = _resolve(model=last_stage, is_draft_worker=False)
        self.assertEqual(
            (info.start_layer, info.end_layer, info.num_effective_layers),
            (39, _NUM_LAYERS, _NUM_LAYERS - 39),
        )

    def test_other_roles_still_reject_split_mtp_target(self):
        last_stage = SimpleNamespace(start_layer=39, end_layer=_NUM_LAYERS)
        for role in ("null", "decode"):
            with (
                self.subTest(role=role),
                get_context().override_server_args(
                    model_path="dummy", disaggregation_mode=role
                ),
                self.assertRaisesRegex(AssertionError, "PP is not compatible"),
            ):
                _resolve(model=last_stage, is_draft_worker=False)

    def test_draft_runner_keeps_its_own_single_layer_range(self):
        # The draft ModelRunner patches pp away, so its model carries no
        # start/end layer and resolves to the nextn layer count.
        for role in ("null", "prefill"):
            with (
                self.subTest(role=role),
                get_context().override_server_args(
                    model_path="dummy", disaggregation_mode=role
                ),
            ):
                info = _resolve(model=SimpleNamespace(), is_draft_worker=True)
            self.assertEqual(
                (info.start_layer, info.end_layer, info.num_effective_layers),
                (0, 1, 1),
            )


if __name__ == "__main__":
    unittest.main()

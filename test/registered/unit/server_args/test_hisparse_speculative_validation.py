"""Keep unsupported speculative layouts outside the HiSparse kernel envelope."""

import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.hisparse_hook import validate_hisparse
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _hisparse_args(**overrides) -> ServerArgs:
    kwargs = {
        "model_path": "dummy",
        "enable_hisparse": True,
        "disable_radix_cache": True,
        "kv_cache_dtype": "fp8_e4m3",
        "disaggregation_mode": "decode",
    }
    args = ServerArgs(**{**kwargs, **overrides})
    # model_config_of hands a fixture-supplied config back untouched; the DSA
    # model-class check reads the architecture name and index_topk.
    args._model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=["GlmMoeDsaForCausalLM"], index_topk=2048
        )
    )
    return args


class TestHiSparseSpeculativeValidation(CustomTestCase):
    def test_accepts_linear_native_mtp_with_four_draft_tokens(self):
        args = _hisparse_args(
            speculative_algorithm="EAGLE",
            speculative_num_steps=3,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=4,
        )
        validate_hisparse(args)

    def test_rejects_ngram_too(self):
        with self.assertRaisesRegex(ValueError, "supports EAGLE/NEXTN"):
            validate_hisparse(_hisparse_args(speculative_algorithm="NGRAM"))

    def test_accepts_hisparse_without_speculation(self):
        validate_hisparse(_hisparse_args())

    def test_rejects_deeper_drafts_and_tree_verify(self):
        for changes, message in [
            (
                {"speculative_num_steps": 5, "speculative_num_draft_tokens": 6},
                "2-4 draft tokens",
            ),
            ({"speculative_eagle_topk": 2}, "eagle_topk=1"),
            ({"disaggregation_mode": "prefill"}, "PD decode"),
            ({"speculative_adaptive": True}, "fixed draft depth"),
            ({"device": "cpu"}, "requires CUDA"),
        ]:
            with self.subTest(changes=changes):
                config = dict(
                    speculative_algorithm="EAGLE",
                    speculative_num_steps=3,
                    speculative_eagle_topk=1,
                    speculative_num_draft_tokens=4,
                )
                config.update(changes)
                with self.assertRaisesRegex(ValueError, message):
                    validate_hisparse(_hisparse_args(**config))

    def test_rejects_an_oversized_index_union(self):
        args = _hisparse_args(
            speculative_algorithm="EAGLE",
            speculative_num_steps=3,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=4,
        )
        args._model_config.hf_config.index_topk = 4096
        with self.assertRaisesRegex(ValueError, "<= 8192"):
            validate_hisparse(args)

    def test_validates_supplied_scratch_capacity_before_runtime_is_installed(self):
        kwargs = dict(
            speculative_algorithm="EAGLE",
            speculative_num_steps=3,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=4,
        )
        validate_hisparse(
            _hisparse_args(
                **kwargs,
                hisparse_config='{"device_buffer_size":8192,"spec_scratch_size":64}'
            )
        )
        with self.assertRaisesRegex(ValueError, ">= 4096"):
            validate_hisparse(
                _hisparse_args(
                    **kwargs,
                    hisparse_config='{"device_buffer_size":4096,"spec_scratch_size":64}'
                )
            )


if __name__ == "__main__":
    unittest.main()

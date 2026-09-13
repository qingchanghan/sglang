"""Reject HiSparse/speculative combinations before unsupported KV handling.

The current coordinator does not implement speculative KV bookkeeping or
multi-query swap-in. These tests cover CPU argument validation only.
"""

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
    def test_rejects_eagle(self):
        args = _hisparse_args(
            speculative_algorithm="EAGLE",
            speculative_num_steps=3,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=4,
        )
        with self.assertRaisesRegex(ValueError, "does not support speculative"):
            validate_hisparse(args)

    def test_rejects_ngram_too(self):
        # Every algorithm goes through spec_prepare_for_decode, not just EAGLE.
        with self.assertRaisesRegex(ValueError, "does not support speculative"):
            validate_hisparse(_hisparse_args(speculative_algorithm="NGRAM"))

    def test_accepts_hisparse_without_speculation(self):
        validate_hisparse(_hisparse_args())


if __name__ == "__main__":
    unittest.main()

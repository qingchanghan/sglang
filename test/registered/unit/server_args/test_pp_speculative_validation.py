"""Pipeline parallelism must accept EAGLE (the draft is hosted on the last PP
stage, which is how a PD prefill ships draft KV) while still rejecting the
overlap schedule. Pure argument validation -- CPU only."""

import unittest

from sglang.srt.arg_groups.validation_hook import check_server_args
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _pp_eagle_args(**overrides) -> ServerArgs:
    kwargs = {
        "model_path": "dummy",
        "served_model_name": "dummy",
        "chunked_prefill_size": 8192,
        "page_size": 1,
        "tp_size": 2,
        "pp_size": 2,
        "disable_overlap_schedule": True,
        "speculative_algorithm": "EAGLE",
        "speculative_num_steps": 1,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 2,
    }
    # Resolve defaults first: the hook reads resolved fields (chunked prefill
    # size, served model name) and is invoked by the launcher after resolution.
    args = ServerArgs(**{**kwargs, **overrides})
    args.resolve_once()
    return args


class TestPipelineParallelSpeculativeValidation(CustomTestCase):
    def test_pp_accepts_eagle(self):
        check_server_args(_pp_eagle_args())

    def test_pp_accepts_eagle_for_disaggregation_prefill(self):
        check_server_args(_pp_eagle_args(disaggregation_mode="prefill"))

    def test_pp_still_rejects_overlap_schedule(self):
        with self.assertRaisesRegex(AssertionError, "overlap schedule"):
            check_server_args(_pp_eagle_args(disable_overlap_schedule=False))


if __name__ == "__main__":
    unittest.main()

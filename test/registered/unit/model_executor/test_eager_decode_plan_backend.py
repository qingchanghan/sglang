"""Eager draft decodes must be planned through the draft worker's multi-step
backend, not the runner's resolved (draft-extend) backend; under DP attention
that planning runs after padding, so pre-planned per-row metadata never
mismatches the padded forward. Pure selection logic -- CPU only."""

import unittest
from types import SimpleNamespace

from sglang.srt.model_executor.runner.eager_runner import EagerRunner
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _StubSpecInput(SpecInput):
    def __init__(self, spec_input_type: SpecInputType):
        super().__init__(spec_input_type)

    def get_spec_adjust_token_coefficient(self):
        return 1, 1


def _runner(draft_attn_backend=None) -> EagerRunner:
    # Bypass EagerRunner.__init__ (needs a live ModelRunner); wire only what
    # _decode_plan_backend reads.
    runner = EagerRunner.__new__(EagerRunner)
    runner.model_runner = SimpleNamespace(
        attn_backend=object(), draft_attn_backend=draft_attn_backend
    )
    return runner


class TestEagerDecodePlanBackend(CustomTestCase):
    def test_draft_input_plans_through_multi_step_wrapper(self):
        wrapper = object()
        runner = _runner(draft_attn_backend=wrapper)
        fb = SimpleNamespace(spec_info=_StubSpecInput(SpecInputType.EAGLE_DRAFT))
        self.assertIs(
            runner._decode_plan_backend(forward_batch=fb, attn_backend=object()),
            wrapper,
        )

    def test_verify_input_keeps_resolved_backend(self):
        # A runner that carries the wrapper must still plan non-draft batches
        # with the resolved backend.
        runner = _runner(draft_attn_backend=object())
        resolved = object()
        fb = SimpleNamespace(spec_info=_StubSpecInput(SpecInputType.EAGLE_VERIFY))
        self.assertIs(
            runner._decode_plan_backend(forward_batch=fb, attn_backend=resolved),
            resolved,
        )

    def test_no_published_wrapper_keeps_resolved_backend(self):
        # Runners that never publish a wrapper (targets, multi-layer EAGLE)
        # are untouched even for draft inputs.
        runner = _runner()
        resolved = object()
        fb = SimpleNamespace(spec_info=_StubSpecInput(SpecInputType.EAGLE_DRAFT))
        self.assertIs(
            runner._decode_plan_backend(forward_batch=fb, attn_backend=resolved),
            resolved,
        )


if __name__ == "__main__":
    unittest.main()

"""The DeepGEMM budget collective must span exactly the ranks that run the
runner: a spec draft hosted on one PP stage agrees over its TP group, while the
target keeps the world group. Pure group selection -- CPU only."""

import unittest
from unittest.mock import patch

from sglang.srt.model_executor.model_runner_components import cuda_graph_setup
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestGraphBudgetGroup(CustomTestCase):
    def test_draft_worker_uses_its_tp_group(self):
        tp_group, world_group = object(), object()
        with (
            patch.object(cuda_graph_setup, "get_tp_group", return_value=tp_group),
            patch.object(cuda_graph_setup, "get_world_group", return_value=world_group),
        ):
            self.assertIs(
                cuda_graph_setup.graph_budget_group(is_draft_worker=True), tp_group
            )

    def test_target_keeps_world_group(self):
        tp_group, world_group = object(), object()
        with (
            patch.object(cuda_graph_setup, "get_tp_group", return_value=tp_group),
            patch.object(cuda_graph_setup, "get_world_group", return_value=world_group),
        ):
            self.assertIs(
                cuda_graph_setup.graph_budget_group(is_draft_worker=False),
                world_group,
            )


if __name__ == "__main__":
    unittest.main()

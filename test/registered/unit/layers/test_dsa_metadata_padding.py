import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.attention.dsa.utils import cal_padded_tokens
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDSAMetadataPadding(CustomTestCase):
    def _calculate(
        self, counts, *, tp, dp_rank=0, max_len=False, cp_v2=False, cp_align=1
    ):
        batch = SimpleNamespace(
            global_num_tokens_cpu=list(counts),
            dp_padding_mode=SimpleNamespace(is_max_len=lambda: max_len),
        )
        with (
            get_parallel().override(
                attn_tp_size=tp, attn_cp_size=1, attn_dp_rank=dp_rank
            ),
            patch("sglang.srt.layers.cp.utils.is_cp_v2_active", return_value=False),
            patch("sglang.srt.layers.cp.utils.enable_cp_v2", return_value=cp_v2),
            patch(
                "sglang.srt.layers.cp.padding.get_cp_padding_align_size",
                return_value=cp_align,
            ),
            patch(
                "sglang.srt.layers.attention.dsa.utils.can_dsa_prefill_cp_round_robin_split",
                return_value=False,
            ),
        ):
            result = cal_padded_tokens(batch)
        self.assertEqual(batch.global_num_tokens_cpu, list(counts))
        return result

    def test_attention_tp_alignment_covers_single_and_uneven_dp_batches(self):
        self.assertEqual(self._calculate([3], tp=2), 4)
        self.assertEqual(self._calculate([3, 5], tp=2, dp_rank=1), 6)
        self.assertEqual(self._calculate([3, 6], tp=4, max_len=True), 8)

    def test_cp_v2_verify_keeps_tp_padding_without_adding_cp_padding(self):
        self.assertEqual(self._calculate([3], tp=2, cp_v2=True, cp_align=8), 4)

    def test_tp_alignment_precedes_cp_alignment(self):
        self.assertEqual(self._calculate([5], tp=4, cp_align=6), 12)


if __name__ == "__main__":
    unittest.main()

"""The probe's index-K byte offsets must follow the paged index buffer layout
(page_size*head_dim K bytes, then page_size fp32 scales per page row), and its
fingerprint must see raw bytes of any dtype. Pure tensor math -- CPU only."""

import unittest

import torch
from sglang.srt.disaggregation.draft_pool_probe import (
    index_k_page_offsets,
    prompt_sample_positions,
    row_fingerprint,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDraftPoolProbe(CustomTestCase):
    def test_index_k_offsets_match_accessor_layout(self):
        # Mirrors index_buf_accessor.GetK.torch_fast: token t of a 64-token page
        # keeps its 128 K bytes at t*128 and its scale after all K bytes.
        page, k_start, scale_start = index_k_page_offsets(
            slot=64 * 3 + 5, page_size=64, head_dim=128
        )
        self.assertEqual((page, k_start, scale_start), (3, 5 * 128, 64 * 128 + 5 * 4))

    def test_fingerprint_reads_raw_bytes_of_non_uint8_rows(self):
        row = torch.zeros(4, dtype=torch.bfloat16)
        row[1] = 1.0
        fp = row_fingerprint(row)
        self.assertEqual(fp["bytes"], 8)
        # bf16 1.0 is 0x3f80: both of its bytes are nonzero, 2 of 8 total.
        self.assertEqual(fp["nonzero_frac"], 0.25)
        self.assertNotEqual(
            fp["blake2b8"],
            row_fingerprint(torch.zeros(4, dtype=torch.bfloat16))["blake2b8"],
        )

    def test_prompt_positions_cover_head_mid_tail_without_duplicates(self):
        self.assertEqual(
            prompt_sample_positions(100000), [0, 1, 50000, 50001, 99998, 99999]
        )
        self.assertEqual(prompt_sample_positions(3), [0, 1, 2])
        self.assertEqual(prompt_sample_positions(0), [])


if __name__ == "__main__":
    unittest.main()

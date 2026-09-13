import unittest

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.dsa.ragged_offsets import pad_ragged_offsets
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestRaggedOffsets(unittest.TestCase):
    def test_real_rows_and_invalid_padding(self):
        indices = torch.tensor([[0, 2, -1]] * 212, dtype=torch.int32)
        offsets = torch.arange(212, dtype=torch.int32) * 100000
        padded = F.pad(indices, (0, 0, 0, 4), value=-1)
        with self.assertRaises(RuntimeError):
            padded + offsets.unsqueeze(1)
        aligned = pad_ragged_offsets(offsets, 216).unsqueeze(1)
        actual = torch.where(padded != -1, padded + aligned, padded)
        expected = torch.where(indices != -1, indices + offsets.unsqueeze(1), indices)
        torch.testing.assert_close(actual[:212], expected)
        self.assertTrue(torch.all(actual[212:] == -1).item())

    def test_matrix_offsets_preserve_rows(self):
        offsets = torch.arange(6, dtype=torch.int64).view(3, 2)
        actual = pad_ragged_offsets(offsets, 5)
        torch.testing.assert_close(actual[:3], offsets)
        self.assertTrue(torch.all(actual[3:] == 0).item())

    def test_equal_shape_and_empty(self):
        for size in (0, 4):
            x = torch.arange(size)
            self.assertIs(pad_ragged_offsets(x, size), x)

    def test_excess_metadata_is_rejected(self):
        with self.assertRaises(ValueError):
            pad_ragged_offsets(torch.arange(5), 4)


if __name__ == "__main__":
    unittest.main()

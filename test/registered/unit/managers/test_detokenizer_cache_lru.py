"""Bounded decode states refresh on indexed access and updates."""

import unittest

from sglang.srt.managers.detokenizer_manager import LimitedCapacityDict
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestLimitedCapacityDictIsLru(CustomTestCase):
    def test_read_refreshes_recency(self):
        cache = LimitedCapacityDict(capacity=3)
        cache["a"], cache["b"], cache["c"] = 1, 2, 3
        self.assertEqual(cache["a"], 1)
        cache["e"] = 5
        self.assertIn("a", cache)
        self.assertNotIn("b", cache)

    def test_updating_an_existing_key_evicts_nothing(self):
        cache = LimitedCapacityDict(capacity=2)
        cache["a"], cache["b"] = 1, 2
        cache["b"] = 3
        self.assertEqual(list(cache), ["a", "b"])
        self.assertEqual(cache["b"], 3)

    def test_update_refreshes_recency(self):
        cache = LimitedCapacityDict(capacity=2)
        cache["a"], cache["b"] = 1, 2
        cache["a"] = 3
        cache["c"] = 4
        self.assertIn("a", cache)
        self.assertNotIn("b", cache)

    def test_capacity_is_still_enforced(self):
        cache = LimitedCapacityDict(capacity=2)
        for key in "abc":
            cache[key] = key
        self.assertEqual(len(cache), 2)
        self.assertNotIn("a", cache)

    def test_missing_read_does_not_change_the_cache(self):
        cache = LimitedCapacityDict(capacity=2)
        cache["a"], cache["b"] = 1, 2
        with self.assertRaises(KeyError):
            cache["missing"]
        self.assertEqual(list(cache), ["a", "b"])


if __name__ == "__main__":
    unittest.main()

"""Duplicate terminal rows must preserve batch alignment without hiding lost state."""

import unittest

import msgspec

from sglang.srt.managers.detokenizer_manager import (
    DetokenizerManager,
    LimitedCapacityDict,
)
from sglang.srt.managers.io_struct import BatchStrOutput, BatchTokenIDOutput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ABORT = {"type": "abort"}


class _FakeTokenizer:
    def decode(self, ids, skip_special_tokens=True, spaces_between_special_tokens=True):
        return "".join(chr(ord("a") + (i % 26)) for i in ids)

    def batch_decode(self, ids_list, **kwargs):
        return [self.decode(ids, **kwargs) for ids in ids_list]


def _manager(capacity=8, batch_decode=False):
    manager = object.__new__(DetokenizerManager)
    manager.vocab_size = 128
    manager.decode_status = LimitedCapacityDict(capacity=capacity)
    manager.disable_tokenizer_batch_decode = not batch_decode
    manager.tokenizer = _FakeTokenizer()
    manager.is_tool_call_parser_gpt_oss = False
    return manager


def _recv(rids, finished, decode_ids=None):
    size = len(rids)
    fields = {
        field.name: None
        for field in msgspec.structs.fields(BatchTokenIDOutput)
        if field.default is msgspec.NODEFAULT
        and field.default_factory is msgspec.NODEFAULT
    }
    fields.update(
        rids=list(rids),
        http_worker_ipcs=[f"ipc-{i}" for i in range(size)],
        decoded_texts=[""] * size,
        decode_ids=decode_ids if decode_ids is not None else [[1, 2, 3] for _ in rids],
        read_offsets=[0] * size,
        finished_reasons=list(finished),
        no_stop_trim=[False] * size,
        skip_special_tokens=[True] * size,
        spaces_between_special_tokens=[True] * size,
        prompt_tokens=list(range(10, 10 + size)),
        completion_tokens=list(range(20, 20 + size)),
        reasoning_tokens=[0] * size,
        cached_tokens=[0] * size,
    )
    return BatchTokenIDOutput(**fields)


class TestDetokenizerDuplicateRid(CustomTestCase):
    def test_unfinished_duplicate_ids_warn_once_per_id(self):
        manager = _manager()
        # This diagnoses the existing collision; it does not assert that two
        # active requests sharing an ID have independent or correct text streams.
        with self.assertLogs(
            "sglang.srt.managers.detokenizer_manager", level="WARNING"
        ) as logs:
            manager._decode_batch_token_id_output(
                _recv(["a", "a", "a", "b", "b"], [None] * 5)
            )
        self.assertEqual(len(logs.records), 2)
        self.assertIn("Duplicate request ID a", logs.records[0].getMessage())
        self.assertIn("Duplicate request ID b", logs.records[1].getMessage())

    def test_terminal_duplicates_do_not_log_twice(self):
        manager = _manager()
        with self.assertLogs(
            "sglang.srt.managers.detokenizer_manager", level="WARNING"
        ) as logs:
            output = manager._decode_batch_token_id_output(
                _recv(["dup", "dup", "dup"], [ABORT] * 3)
            )
        self.assertEqual(output, ["bcd", "", ""])
        self.assertEqual(len(logs.records), 1)

    def test_same_rid_twice_in_one_batch_does_not_raise(self):
        for batch_decode in (False, True):
            with self.subTest(batch_decode=batch_decode):
                manager = _manager(batch_decode=batch_decode)
                output = manager._decode_batch_token_id_output(
                    _recv(["duplicate-abort", "duplicate-abort"], [ABORT, ABORT])
                )
                self.assertEqual(output, ["bcd", ""])
                self.assertNotIn("duplicate-abort", manager.decode_status)

    def test_batch_str_output_preserves_text_and_metadata_alignment(self):
        for batch_decode in (False, True):
            with self.subTest(batch_decode=batch_decode):
                manager = _manager(batch_decode=batch_decode)
                recv = _recv(
                    ["a", "dup", "dup", "b"],
                    [None, ABORT, ABORT, None],
                    [[0], [1], [2], [3]],
                )
                output = manager.handle_batch_token_id_out(recv)
                self.assertIsInstance(output, BatchStrOutput)
                self.assertEqual(output.output_strs, ["a", "b", "", "d"])
                for name in (
                    "rids",
                    "http_worker_ipcs",
                    "finished_reasons",
                    "prompt_tokens",
                    "completion_tokens",
                ):
                    self.assertEqual(getattr(output, name), getattr(recv, name))

    def test_multiple_terminal_duplicates_leave_one_slot_each(self):
        manager = _manager()
        output = manager._decode_batch_token_id_output(
            _recv(["dup", "dup", "dup", "other"], [ABORT, ABORT, ABORT, None])
        )
        self.assertEqual(output, ["bcd", "", "", "bcd"])
        self.assertNotIn("dup", manager.decode_status)
        self.assertIn("other", manager.decode_status)

    def test_distinct_streaming_updates_keep_incremental_text(self):
        for batch_decode in (False, True):
            with self.subTest(batch_decode=batch_decode):
                manager = _manager(batch_decode=batch_decode)
                output = []
                with self.assertNoLogs(
                    "sglang.srt.managers.detokenizer_manager", level="WARNING"
                ):
                    for ids, finished in [([1, 2], None), ([3], None), ([4], ABORT)]:
                        output.extend(
                            manager._decode_batch_token_id_output(
                                _recv(["stream"], [finished], [ids])
                            )
                        )
                self.assertEqual(output, ["bc", "d", "e"])
                self.assertNotIn("stream", manager.decode_status)

    def test_finished_tracking_does_not_leak_into_the_next_batch(self):
        manager = _manager()
        manager._decode_batch_token_id_output(_recv(["reuse", "reuse"], [ABORT, ABORT]))
        output = manager._decode_batch_token_id_output(_recv(["reuse"], [None], [[4]]))
        self.assertEqual(output, ["e"])
        self.assertIn("reuse", manager.decode_status)

    def test_real_capacity_eviction_is_not_silently_converted_to_empty_text(self):
        manager = _manager(capacity=1)
        with self.assertRaisesRegex(RuntimeError, "Decode status not found"):
            manager._decode_batch_token_id_output(_recv(["a", "b"], [None, None]))

    def test_unfinished_row_after_terminal_is_not_silently_dropped(self):
        manager = _manager()
        with self.assertRaisesRegex(RuntimeError, "Decode status not found"):
            manager._decode_batch_token_id_output(_recv(["dup", "dup"], [ABORT, None]))


if __name__ == "__main__":
    unittest.main()

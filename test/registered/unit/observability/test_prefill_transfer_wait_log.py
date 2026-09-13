"""The prefill request-time log must report the wait between the last chunk
finishing and the decode side acknowledging the KV transfer. forward_duration
runs to completion and so folds compute into the wait, and transfer_speed is
derived from the transfer-queue entry, so without this field a slow transfer
path cannot be told apart from slow prefill. Pure formatting -- CPU only."""

import unittest

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.observability.req_time_stats import SchedulerReqTimeStats
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _prefill_stats(*, kv_transfer_finish_time: float) -> SchedulerReqTimeStats:
    stats = SchedulerReqTimeStats()
    stats.disagg_mode = DisaggregationMode.PREFILL
    stats.prefill_bootstrap_queue_entry_time = 100.0
    stats.wait_queue_entry_time = 100.1
    stats.forward_entry_time = 100.2
    stats.prefill_finished_time = 100.5
    stats.prefill_kv_transfer_finish_time = kv_transfer_finish_time
    stats.completion_time = 101.75
    return stats


class TestPrefillTransferWaitLog(CustomTestCase):
    def test_reports_last_chunk_to_transfer_ack(self):
        line = _prefill_stats(kv_transfer_finish_time=101.7).convert_to_duration()
        self.assertIn("transfer_wait=1200.00ms", line)

    def test_unstamped_transfer_finish_reports_zero_not_negative(self):
        # A request that failed or was aborted before the ack never stamps the
        # finish time; the sentinel must not turn into a negative duration.
        line = _prefill_stats(kv_transfer_finish_time=0.0).convert_to_duration()
        self.assertIn("transfer_wait=0.00ms", line)


if __name__ == "__main__":
    unittest.main()

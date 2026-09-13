"""The PP output ring must never have every rank posting its send before its
recv in one round: a NCCL send completes only when the peer posts the matching
recv, and that recv sits behind the peer's own send, so an all-senders cycle
deadlocks (seen as a chunked-prefill hand-off with MTP output dicts). Pure
ordering math on the ring wait graph -- CPU only."""

import unittest

from sglang.srt.managers.scheduler_pp_mixin import pp_output_exchange_send_first
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _ring_deadlocks(pp_size: int, send_first) -> bool:
    # Each rank sends to the next rank and receives from the previous one. A
    # send-first rank blocks until its successor posts a recv; the successor
    # posts that recv only once its own first op completes. The round deadlocks
    # iff following "waits on" from any rank loops back to it.
    def blocker(rank: int):
        nxt = (rank + 1) % pp_size
        # rank waits on nxt only when both put their send first.
        return nxt if send_first(rank) and send_first(nxt) else None

    for start in range(pp_size):
        seen = set()
        rank = start
        while rank is not None and rank not in seen:
            seen.add(rank)
            rank = blocker(rank)
        if rank is not None:
            return True
    return False


class TestPPOutputExchangeOrder(CustomTestCase):
    def test_all_send_first_deadlocks_any_ring(self):
        for pp_size in range(2, 6):
            self.assertTrue(_ring_deadlocks(pp_size, lambda rank: True), pp_size)

    def test_parity_order_is_deadlock_free_for_every_ring_size(self):
        for pp_size in range(2, 9):
            self.assertFalse(
                _ring_deadlocks(
                    pp_size, lambda rank: pp_output_exchange_send_first(pp_rank=rank)
                ),
                pp_size,
            )

    def test_parity_alternates(self):
        self.assertTrue(pp_output_exchange_send_first(pp_rank=0))
        self.assertFalse(pp_output_exchange_send_first(pp_rank=1))
        self.assertTrue(pp_output_exchange_send_first(pp_rank=2))


if __name__ == "__main__":
    unittest.main()

"""Source lifetimes require native completion and ACKs from every D reader."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfa_recent128.pd_lifetime import PublicationTracker, PublicationLifetimeError


class PublicationLifetimeTests(unittest.TestCase):
    def test_two_readers_first_ack_and_native_done_cannot_free_second_reader(self):
        tracker = PublicationTracker(2)
        tracker.add("p")
        tracker.subscribe("p", "engine-d1:port:r", "nonce")
        tracker.subscribe("p", "engine-d2:port:r", "nonce")
        self.assertTrue(tracker.ack("p", "engine-d1:port:r", "nonce"))
        tracker.native_done("p")
        self.assertFalse(tracker.releasable("p"))
        with self.assertRaises(PublicationLifetimeError):
            tracker.release("p")
        self.assertTrue(tracker.ack("p", "engine-d2:port:r", "nonce"))
        self.assertTrue(tracker.releasable("p"))
        self.assertTrue(tracker.release("p"))
        self.assertFalse(tracker.releasable("p"))

    def test_native_done_before_ack_and_unused_source(self):
        tracker = PublicationTracker(2)
        tracker.add("p")
        tracker.subscribe("p", "d", "nonce")
        tracker.native_done("p")
        self.assertFalse(tracker.releasable("p"))
        tracker.ack("p", "d", "nonce")
        self.assertTrue(tracker.releasable("p"))
        tracker.add("unused")
        self.assertFalse(tracker.releasable("unused"))
        tracker.native_done("unused")
        self.assertTrue(tracker.releasable("unused"))
        tracker.release("unused")

    def test_ack_before_native_done_is_retained(self):
        tracker = PublicationTracker(1)
        tracker.add("p")
        tracker.subscribe("p", "d", "n")
        tracker.ack("p", "d", "n")
        self.assertFalse(tracker.releasable("p"))
        tracker.native_done("p")
        self.assertTrue(tracker.releasable("p"))

    def test_wrong_nonce_unknown_receiver_and_exact_ack_retry(self):
        tracker = PublicationTracker(1)
        tracker.add("p")
        self.assertTrue(tracker.subscribe("p", "d", "n"))
        self.assertFalse(tracker.subscribe("p", "d", "n"))
        with self.assertRaises(PublicationLifetimeError):
            tracker.subscribe("p", "d", "changed")
        self.assertFalse(tracker.ack("p", "other-d", "n"))
        self.assertFalse(tracker.ack("p", "d", "wrong"))
        self.assertTrue(tracker.ack("p", "d", "n"))
        self.assertTrue(tracker.ack("p", "d", "n"))
        tracker.native_done("p")
        tracker.release("p")
        self.assertTrue(tracker.ack("p", "d", "n"))
        self.assertFalse(tracker.ack("p", "other-d", "n"))
        self.assertFalse(tracker.ack("p", "d", "wrong"))
        with self.assertRaisesRegex(PublicationLifetimeError, "unknown or released"):
            tracker.subscribe("p", "late-d", "n")

    def test_reused_request_cannot_accept_previous_lifecycle_ack(self):
        tracker = PublicationTracker(1)
        tracker.add("p")
        tracker.subscribe("p", "d", "old")
        tracker.ack("p", "d", "old")
        tracker.native_done("p")
        tracker.release("p")
        self.assertTrue(tracker.ack("p", "d", "old"))
        tracker.add("p")
        tracker.subscribe("p", "d", "new")
        tracker.native_done("p")
        self.assertFalse(tracker.ack("p", "d", "old"))
        self.assertFalse(tracker.releasable("p"))
        tracker.ack("p", "d", "new")
        self.assertTrue(tracker.releasable("p"))

    def test_bounded_active_readers_and_released_ack_tombstones(self):
        tracker = PublicationTracker(1, max_recent_acks=2, max_receivers_per_request=1)
        for index in range(3):
            rid = "p" + str(index)
            tracker.add(rid)
            self.assertFalse(tracker.add(rid))
            with self.assertRaisesRegex(PublicationLifetimeError, "full"):
                tracker.add("other")
            tracker.subscribe(rid, "d", "n")
            with self.assertRaisesRegex(PublicationLifetimeError, "receiver limit"):
                tracker.subscribe(rid, "another-d", "n")
            tracker.ack(rid, "d", "n")
            tracker.native_done(rid)
            tracker.release(rid)
        self.assertFalse(tracker.ack("p0", "d", "n"))
        self.assertTrue(tracker.ack("p1", "d", "n"))
        self.assertTrue(tracker.ack("p2", "d", "n"))
        self.assertFalse(tracker.native_done("unknown"))
        self.assertFalse(tracker.release("unknown"))


if __name__ == "__main__":
    unittest.main()

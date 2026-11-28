"""Bounded source-publication ownership across native completion and D readers.

Every D worker subscribes before receiving a descriptor. Its exact ACK only
releases that subscription; source memory remains owned until native completion
and every issued subscription have completed. The bridge must serialize its
external pool release with subscribe/ACK/native-done handling under its lock.
"""
from collections import OrderedDict
from dataclasses import dataclass, field
import threading


class PublicationLifetimeError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise PublicationLifetimeError(message)


def _identity(value, name):
    _require(type(value) in (str, int) and value != "", name + " must be a stable string or integer")
    return value


def _nonce(value):
    _require(type(value) is str and bool(value), "Publication nonce must be a nonempty string")
    return value


@dataclass
class _Reader:
    nonce: str
    acknowledged: bool = False


@dataclass
class _Request:
    native_done: bool = False
    readers: dict = field(default_factory=dict)


class PublicationTracker:
    """Track active requests and retain a bounded cache of released exact ACKs.

    ``add`` is idempotent for an active request and never resets its readers.
    ``subscribe`` is idempotent for the same receiver/nonce; a changed nonce
    for that receiver is a protocol error. ``ack`` returns False for unknown
    receivers or nonces, and True for an exact ACK or its exact retry.
    """
    def __init__(self, max_active, max_recent_acks=256,
                 max_receivers_per_request=256):
        for name, value, minimum in (("max_active", max_active, 1),
                ("max_recent_acks", max_recent_acks, 0),
                ("max_receivers_per_request", max_receivers_per_request, 1)):
            _require(type(value) is int and value >= minimum,
                     name + " must be an integer >= " + str(minimum))
        self.max_active = max_active
        self.max_recent_acks = max_recent_acks
        self.max_receivers_per_request = max_receivers_per_request
        self._active = {}
        self._recent_acks = OrderedDict()
        self._lock = threading.RLock()

    def add(self, request_id):
        _identity(request_id, "request_id")
        with self._lock:
            if request_id in self._active:
                return False
            _require(len(self._active) < self.max_active,
                     "Source publication tracker is full")
            # A reused request identity starts a fresh lifecycle. Earlier
            # released ACKs cannot acknowledge any part of that lifecycle.
            for key in tuple(self._recent_acks):
                if key[0] == request_id:
                    del self._recent_acks[key]
            self._active[request_id] = _Request()
            return True

    def subscribe(self, request_id, receiver_id, nonce):
        _identity(request_id, "request_id")
        _identity(receiver_id, "receiver_id")
        _nonce(nonce)
        with self._lock:
            request = self._active.get(request_id)
            _require(request is not None, "Cannot subscribe to an unknown or released publication")
            reader = request.readers.get(receiver_id)
            if reader is not None:
                _require(reader.nonce == nonce, "Receiver subscription changed its publication nonce")
                return False
            _require(len(request.readers) < self.max_receivers_per_request,
                     "Source publication receiver limit reached")
            request.readers[receiver_id] = _Reader(nonce)
            return True

    def ack(self, request_id, receiver_id, nonce):
        _identity(request_id, "request_id")
        _identity(receiver_id, "receiver_id")
        _nonce(nonce)
        with self._lock:
            request = self._active.get(request_id)
            if request is None:
                return (request_id, receiver_id, nonce) in self._recent_acks
            reader = request.readers.get(receiver_id)
            if reader is None or reader.nonce != nonce:
                return False
            reader.acknowledged = True
            return True

    def native_done(self, request_id):
        _identity(request_id, "request_id")
        with self._lock:
            request = self._active.get(request_id)
            if request is None:
                return False
            request.native_done = True
            return True

    @staticmethod
    def _ready(request):
        return request.native_done and all(reader.acknowledged for reader in request.readers.values())

    def releasable(self, request_id):
        _identity(request_id, "request_id")
        with self._lock:
            request = self._active.get(request_id)
            return request is not None and self._ready(request)

    def release(self, request_id):
        """Call after the bridge has successfully released source memory."""
        _identity(request_id, "request_id")
        with self._lock:
            request = self._active.get(request_id)
            if request is None:
                return False
            _require(self._ready(request), "Source publication still has native work or unacknowledged readers")
            for receiver_id, reader in request.readers.items():
                key = (request_id, receiver_id, reader.nonce)
                self._recent_acks[key] = None
                self._recent_acks.move_to_end(key)
            while len(self._recent_acks) > self.max_recent_acks:
                self._recent_acks.popitem(last=False)
            del self._active[request_id]
            return True

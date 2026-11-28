"""Worker-owned recent128 PD tails at a scheduler-committed boundary.

This module is a request/transfer-boundary helper, never a forward hook. The
caller supplies the actual committed end and a real device completion object.
Tag validation reads copied tags on the host here; its synchronization and
transfer costs belong to measured request latency. It must not run in Graph
capture or inside an attention layer.
"""
from dataclasses import replace

from .serving_state import ServingStateError, TailSnapshot


def _require(condition, message):
    if not condition:
        raise ServingStateError(message)


def _complete(completion):
    """Complete actual device work, rather than trusting a Python return."""
    _require(completion is not None, "PD transfer needs an actual device completion")
    wait = getattr(completion, "wait", None)
    synchronize = getattr(completion, "synchronize", None)
    _require(callable(wait) and callable(synchronize),
             "PD completion must provide wait() and synchronize()")
    wait()
    synchronize()


def _host_values(value):
    # Deliberately confined to this transfer-boundary module. In particular,
    # forward/bind paths must not call this validation helper.
    return value.detach().cpu().tolist()


def validate_tail(state, snapshot):
    """Validate every source tag and layer before admitting received BF16."""
    owner = getattr(state, "pool_owner", state)
    _require(isinstance(snapshot, TailSnapshot), "Expected an actual BF16 TailSnapshot")
    _require(type(snapshot.end) is int and snapshot.end >= 0 and
             type(snapshot.start) is int and snapshot.start == max(0, snapshot.end - 128),
             "PD snapshot must contain the exact committed last-128 interval")
    _require(type(snapshot.source_epoch) is int and snapshot.source_epoch > 0,
             "PD snapshot needs an actual positive source epoch")
    _require(set(snapshot.layers) == set(owner.layers) and bool(owner.layers),
             "PD snapshot must cover every local attention layer")
    count = snapshot.end - snapshot.start
    expected_positions = list(range(snapshot.start, snapshot.end))
    expected_epochs = [snapshot.source_epoch] * count
    for tail in snapshot.layers.values():
        values = (tail.latent, tail.rope, tail.position_tags, tail.epoch_tags)
        for value, shape in zip(values, ((count, 512), (count, 64), (count,), (count,))):
            _require(tuple(value.shape) == shape and value.device == owner.device,
                     "PD tail payload must match the local layer shape and device")
        t = owner.torch
        _require(tail.latent.dtype == t.bfloat16 and tail.rope.dtype == t.bfloat16 and
                 tail.position_tags.dtype == t.int64 and tail.epoch_tags.dtype == t.int64,
                 "PD payload must preserve BF16 values and INT64 tags")
        _require(_host_values(tail.position_tags) == expected_positions,
                 "PD tail has missing or stale logical position tags")
        _require(_host_values(tail.epoch_tags) == expected_epochs,
                 "PD tail has missing or stale source epoch tags")
    return snapshot


def export_tail(state, request_id, *, committed_end, completion):
    """Export after real producer completion, using a scheduler-supplied end."""
    owner = getattr(state, "pool_owner", state)
    _require(type(committed_end) is int and committed_end >= 0,
             "PD export needs an actual scheduler-committed end")
    _require(request_id in owner.requests, "PD export request has no local lease")
    lease = owner.requests[request_id]
    _require(lease.speculation is None, "Resolve speculative acceptance before PD export")
    _complete(completion)
    # Save/restore so a failed source tag check cannot publish a false commit.
    previous_end = lease.committed_end
    try:
        owner.commit_request(request_id, end=committed_end)
        snapshot = owner.export_tail(request_id, end=committed_end)
        validate_tail(owner, snapshot)
    except Exception:
        lease.committed_end = previous_end
        raise
    # The producer event precedes index_select snapshot copies. Do not label
    # that event as completion of those later copies; validation above waited
    # for copied tag data, and receiver must supply its own receive completion.
    return replace(snapshot, completion=None)


def _check_receive_lease(owner, request_id, end):
    lease = owner.requests.get(request_id)
    if lease is None:
        _require(bool(owner._free), "PD receive request ring pool exhausted")
        return None
    _require(lease.speculation is None, "Resolve speculation before PD receive")
    if lease.committed_end != end:
        _require(lease.committed_end == 0 and lease.tail_source in
                 ("new_request", "tail_import_required"),
                 "PD receive cannot replace a different committed request history")
        # Merely having no CPU commit does not establish that a reserved lease
        # is empty: normal forward writes tags without automatic host commits.
        _require(all(lease.epoch not in _host_values(ring.epoch_tags[lease.bank])
                     for ring in owner.layers.values()),
                 "PD receive cannot replace a lease containing producer rows")
    return lease


def _install_tail(owner, request_id, snapshot, *, completion, source):
    lease = owner.requests.get(request_id)
    if lease is None:
        lease = owner.acquire_request(request_id, initial_position=snapshot.end)
    else:
        # A worker may reserve a lease at zero before its BF16 tail arrives.
        # Move its committed boundary, retaining the existing bank and epoch.
        lease.committed_end = snapshot.end
    owner.import_tail(request_id, snapshot, source=source, completion=completion)
    return lease


def receive_tail(state, request_id, snapshot, *, completion, source="pd"):
    """Import into an owner before metadata bind, preserving existing epochs."""
    owner = getattr(state, "pool_owner", state)
    _require(source in ("pd", "prefix", "resume"), "Unknown tail import source")
    _complete(completion)
    validate_tail(owner, snapshot)
    _check_receive_lease(owner, request_id, snapshot.end)
    return _install_tail(owner, request_id, snapshot, completion=completion, source=source)


class WorkerTailRegistry:
    """Local pool owners; routing lanes share an owner and are deduplicated."""
    def __init__(self, states=()):
        self.states = []
        for state in states:
            self.register(state)

    def register(self, state):
        owner = getattr(state, "pool_owner", state)
        if not any(existing is owner for existing in self.states):
            self.states.append(owner)
        return owner

    def request_identity(self, request_id):
        return tuple((tuple(state.layers), lease.bank, lease.epoch, lease.committed_end)
                     for state in self.states
                     if (lease := state.requests.get(request_id)) is not None)

    def export_request(self, request_id, *, committed_end, completion):
        _require(bool(self.states), "No recent128 local pool owner is registered")
        _require(all(request_id in state.requests for state in self.states),
                 "PD export request must exist in every registered local pool owner")
        return tuple(export_tail(state, request_id, committed_end=committed_end,
                                 completion=completion) for state in self.states)

    def receive_request(self, request_id, snapshots, *, completion, source="pd"):
        if isinstance(snapshots, TailSnapshot):
            snapshots = (snapshots,)
        snapshots = tuple(snapshots)
        _require(source in ("pd", "prefix", "resume"), "Unknown tail import source")
        _require(bool(self.states), "No recent128 local pool owner is registered")
        _require(all(isinstance(snapshot, TailSnapshot) for snapshot in snapshots),
                 "PD receive requires actual BF16 TailSnapshots")
        _require(len({snapshot.end for snapshot in snapshots}) == 1,
                 "PD tails must share one scheduler-committed request end")
        pairs = []
        for state in self.states:
            matches = [snapshot for snapshot in snapshots
                       if set(snapshot.layers) == set(state.layers)]
            _require(len(matches) == 1, "PD receive needs exactly one tail for every local pool owner")
            pairs.append((state, matches[0]))
        _require(all(any(snapshot is selected for _, selected in pairs) for snapshot in snapshots),
                 "PD receive contains an unexpected local layer set")
        # Validate all owners before the first import to avoid partial admission
        # on malformed payloads from another local layer set.
        _complete(completion)
        for state, snapshot in pairs:
            validate_tail(state, snapshot)
            _check_receive_lease(state, request_id, snapshot.end)
        return tuple(_install_tail(state, request_id, snapshot, completion=completion,
                                   source=source) for state, snapshot in pairs)

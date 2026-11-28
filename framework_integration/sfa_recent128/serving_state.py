"""Request-owned BF16 sidecars and step-owned device routing for serving.

The runner calls ``bind_step`` once after preparing the *current* device
positions/query boundaries (including speculative corrections). Layers only
obtain cached ``RingBinding`` views. No device value is read by this module.
Request IDs and initial positions come from scheduler state, never C4 slots.

Lifecycle copies run on the caller's current stream. A cross-stream/PD import
must provide an event whose ``wait()`` orders that stream before the copy.
Readers must check logical position AND epoch tags and apply causal masking;
rejected speculative rows intentionally remain tagged future rows.
"""
from dataclasses import dataclass, field
import heapq
from collections.abc import Mapping

from .prolog_provider import RingBinding


class ServingStateError(RuntimeError):
    pass


def _positive(name, value, minimum=1):
    if type(value) is not int or value < minimum:
        raise ServingStateError(name + " must be an integer >= " + str(minimum))
    return value


def _require(condition, message):
    if not condition:
        raise ServingStateError(message)


@dataclass
class RequestLease:
    request_id: object
    bank: int
    epoch: int
    committed_end: int = 0
    tail_ready: bool = True
    tail_source: str = "new_request"
    speculation: tuple | None = None


@dataclass(frozen=True)
class LayerRing:
    latent: object
    rope: object
    position_tags: object
    epoch_tags: object


@dataclass(frozen=True)
class LayerTail:
    latent: object
    rope: object
    position_tags: object
    epoch_tags: object


@dataclass(frozen=True)
class TailSnapshot:
    start: int
    end: int
    source_epoch: int
    layers: dict
    completion: object = None


@dataclass(frozen=True)
class BoundStep:
    token_count: int
    request_ids: tuple
    token_ring_slots: object
    token_positions: object
    token_request_epochs: object
    token_valid: object
    request_banks: object
    request_epochs: object
    leases: tuple = field(repr=False)


class ServingState:
    def __init__(self, *, request_pool_size, max_tokens, max_query_per_request,
                 device, torch_module=None, request_table_capacity=None):
        if torch_module is None:
            import torch as torch_module
        self.torch = torch_module
        self.device = torch_module.device(device)
        self.request_pool_size = _positive("request_pool_size", request_pool_size)
        self.max_tokens = _positive("max_tokens", max_tokens)
        self.max_query_per_request = _positive("max_query_per_request", max_query_per_request)
        self.request_table_capacity = _positive("request_table_capacity",
            request_pool_size if request_table_capacity is None else request_table_capacity)
        _require(self.request_table_capacity >= request_pool_size,
                 "Request table capacity must cover the physical request pool")
        self.capacity = ((128 + max_query_per_request - 1 + 127) // 128) * 128
        self.layers = {}
        self.requests = {}
        self._epochs = [0] * request_pool_size
        self._free = list(range(request_pool_size))
        self._views = {}
        self._bindings = {}
        self.step = None
        self.pool_owner = self
        self._routing_lanes = {}
        t = self.torch
        device = self.device
        p, n = self.request_table_capacity, max_tokens
        self.request_banks = t.full((p,), -1, dtype=t.int32, device=device)
        self.request_epochs = t.full((p,), -1, dtype=t.int64, device=device)
        # Scheduler identity is CPU data already; these fixed staging tensors
        # avoid allocating a new transfer tensor for every step/layer.
        self._banks_host = t.full((p,), -1, dtype=t.int32, device="cpu")
        self._epochs_host = t.full((p,), -1, dtype=t.int64, device="cpu")
        self.token_ring_slots = t.full((n,), -1, dtype=t.int32, device=device)
        self.token_positions = t.full((n,), -1, dtype=t.int64, device=device)
        self.token_request_epochs = t.full((n,), -1, dtype=t.int64, device=device)
        self.token_valid = t.full((n,), False, dtype=t.bool, device=device)
        self._indices = t.empty((n,), dtype=t.int64, device=device)
        self._rows = t.arange(n, dtype=t.int64, device=device)
        self._mask = t.empty((n,), dtype=t.bool, device=device)
        self._other_mask = t.empty((n,), dtype=t.bool, device=device)

    def register_layer(self, layer_key):
        """Allocate before warmup/capture; registration is idempotent."""
        if self.pool_owner is not self:
            return self.pool_owner.register_layer(layer_key)
        if layer_key in self.layers:
            return self.layers[layer_key]
        _require(not self.requests, "Register all layers before admitting requests")
        t, p, c = self.torch, self.request_pool_size, self.capacity
        ring = LayerRing(
            t.empty((p, c, 512), dtype=t.bfloat16, device=self.device),
            t.empty((p, c, 64), dtype=t.bfloat16, device=self.device),
            t.full((p, c), -1, dtype=t.int64, device=self.device),
            t.full((p, c), -1, dtype=t.int64, device=self.device))
        self.layers[layer_key] = ring
        for count in self._views:
            self._make_binding(layer_key, count)
        for lane in self._routing_lanes.values():
            for count in lane._views:
                lane._make_binding(layer_key, count)
        return ring

    def routing_lane(self, key, *, allow_create=True):
        """Independent static routes for one builder/draft execution lane.

        Physical layer rings, stable leases and epochs remain shared. Create
        lanes during warmup, then pass allow_create=False during capture;
        creating a new lane allocates device memory and is never a replay op.
        """
        owner = self.pool_owner
        if key in owner._routing_lanes:
            return owner._routing_lanes[key]
        _require(allow_create, "Routing lane was not preallocated before Graph capture")
        lane = ServingState(request_pool_size=owner.request_pool_size,
            max_tokens=owner.max_tokens,
            max_query_per_request=owner.max_query_per_request,
            device=owner.device, torch_module=owner.torch,
            request_table_capacity=owner.request_table_capacity)
        lane.pool_owner = owner
        lane.layers = owner.layers
        lane.requests = owner.requests
        lane._epochs = owner._epochs
        lane._free = owner._free
        lane._routing_lanes = owner._routing_lanes
        owner._routing_lanes[key] = lane
        return lane

    def _make_binding(self, layer_key, count):
        ring, v = self.layers[layer_key], self._views[count]
        binding = RingBinding(ring.latent, ring.rope, ring.position_tags,
                              ring.epoch_tags, v[0], v[1], v[2], v[3],
                              self.max_query_per_request)
        self._bindings[layer_key, count] = binding
        return binding

    def prepare_token_count(self, count):
        """Cache graph-bucket views outside a layer's captured execution."""
        _positive("token_count", count)
        _require(count <= self.max_tokens, "token_count exceeds preallocated capacity")
        if count not in self._views:
            self._views[count] = tuple(x[:count] for x in (
                self.token_ring_slots, self.token_positions,
                self.token_request_epochs, self.token_valid))
            for layer_key in self.layers:
                self._make_binding(layer_key, count)
        return self._views[count]

    def acquire_request(self, request_id, *, initial_position=0):
        _positive("initial_position", initial_position, 0)
        try:
            existing = self.requests.get(request_id)
        except TypeError as error:
            raise ServingStateError("Scheduler request IDs must be stable hashable values") from error
        if existing is not None:
            return existing
        _require(request_id is not None, "None is not a scheduler request ID")
        _require(bool(self._free), "Request ring pool exhausted; release finished requests explicitly")
        bank = heapq.heappop(self._free)
        self._epochs[bank] += 1
        lease = RequestLease(request_id, bank, self._epochs[bank], initial_position,
                             initial_position == 0,
                             "new_request" if initial_position == 0 else "tail_import_required")
        self.requests[request_id] = lease
        return lease

    @staticmethod
    def _wait(completion):
        if completion is not None:
            wait = getattr(completion, "wait", None)
            _require(callable(wait), "Completion must provide a stream-ordering wait()")
            wait()

    def release_request(self, request_id, *, completion=None):
        """Call for scheduler-finished requests, including evicted requests."""
        lease = self.requests.get(request_id)
        if lease is None:
            return False
        self._wait(completion)
        del self.requests[request_id]
        heapq.heappush(self._free, lease.bank)
        return True

    def bind_step(self, request_ids, token_to_req=None, positions=None, *,
                  query_start_loc=None, valid=None, num_actual_tokens=None,
                  request_starts=None, token_count=None):
        """Update device routes once per runner step, before replay/forward.

        ``request_starts`` is mandatory for newly admitted requests. A new
        nonzero start requires prefix/PD ``import_tail`` before binding. For
        existing requests corrected GPU positions are authoritative, so MTP
        does not read accepted counts back to CPU or impose monotonic starts.
        ``query_start_loc`` derives routes when the ordinary SFA builder does
        not populate its optional offload-only ``token_to_req`` field.
        """
        ids = tuple(request_ids)
        _require(len(ids) <= self.request_pool_size and len(set(ids)) == len(ids),
                 "Step request IDs must be unique and fit the pool")
        _require(positions is not None, "Current corrected device positions are required")
        count = int(positions.shape[0]) if token_count is None else token_count
        self.prepare_token_count(count)
        _require(tuple(positions.shape) == (count,), "positions must be [token_count]")
        _require(getattr(positions, "device", None) == self.device,
                 "positions must remain on the configured device")
        if request_starts is not None and not isinstance(request_starts, Mapping):
            _require(len(request_starts) == len(ids), "request_starts length mismatch")
            request_starts = dict(zip(ids, request_starts))
        # Validate the whole scheduler plan before admitting/mutating leases.
        new_ids = [request_id for request_id in ids if request_id not in self.requests]
        _require(len(new_ids) <= len(self._free), "Request ring pool exhausted")
        for request_id in new_ids:
            _require(request_starts is not None and request_id in request_starts,
                     "New requests require scheduler initial positions; do not infer them from C4 slots")
            start = _positive("request initial position", request_starts[request_id], 0)
            _require(start == 0, "Nonzero request start needs acquire_request + prefix/PD import_tail first")
        for request_id in ids:
            lease = self.requests.get(request_id)
            _require(lease is None or lease.tail_ready,
                     "Request BF16 tail is not ready; C4 readiness is insufficient")
        leases = tuple(self.acquire_request(request_id,
                       initial_position=(request_starts[request_id] if request_starts is not None else 0))
                       for request_id in ids)
        self._banks_host.fill_(-1)
        self._epochs_host.fill_(-1)
        for row, lease in enumerate(leases):
            self._banks_host[row] = lease.bank
            self._epochs_host[row] = lease.epoch
        self.request_banks.copy_(self._banks_host)
        self.request_epochs.copy_(self._epochs_host)
        t = self.torch
        banks, pos, epochs, active = self._views[count]
        idx, mask, other = self._indices[:count], self._mask[:count], self._other_mask[:count]
        if token_to_req is None:
            _require(query_start_loc is not None and
                     tuple(query_start_loc.shape) == (len(ids) + 1,),
                     "Provide device token_to_req or current device query_start_loc[B+1]")
            _require(query_start_loc.device == self.device, "query_start_loc must remain on device")
            t.searchsorted(query_start_loc[1:], self._rows[:count], right=True, out=idx)
        else:
            _require(tuple(token_to_req.shape) == (count,) and token_to_req.device == self.device,
                     "token_to_req must remain device [token_count]")
            idx.copy_(token_to_req)
        t.ge(idx, 0, out=active)
        t.lt(idx, len(ids), out=mask)
        active.logical_and_(mask)
        pos.copy_(positions)
        t.ge(pos, 0, out=mask)
        active.logical_and_(mask)
        if valid is not None:
            _require(tuple(valid.shape) == (count,) and valid.device == self.device and
                     valid.dtype == t.bool, "valid must be a device BOOL [token_count]")
            active.logical_and_(valid)
        if num_actual_tokens is not None:
            _positive("num_actual_tokens", num_actual_tokens, 0)
            _require(num_actual_tokens <= count, "num_actual_tokens exceeds padded token count")
            t.lt(self._rows[:count], num_actual_tokens, out=mask)
            active.logical_and_(mask)
        elif valid is None:
            _require(token_to_req is None,
                     "Direct token_to_req routes need device valid or scheduler num_actual_tokens")
        idx.clamp_(0, self.request_table_capacity - 1)
        t.index_select(self.request_banks, 0, idx, out=banks)
        t.index_select(self.request_epochs, 0, idx, out=epochs)
        other.copy_(active).logical_not_()
        banks.masked_fill_(other, -1)
        epochs.masked_fill_(other, -1)
        pos.masked_fill_(other, -1)
        self.step = BoundStep(count, ids, banks, pos, epochs, active,
                              self.request_banks, self.request_epochs, leases)
        return self.step

    def binding(self, layer_key, token_rows=None):
        _require(self.step is not None, "Runner must bind the current step before a layer executes")
        count = self.step.token_count if token_rows is None else token_rows
        _require(count == self.step.token_count, "Layer rows differ from the runner's bound step")
        try:
            return self._bindings[layer_key, count]
        except KeyError as error:
            raise ServingStateError("Layer/bucket was not preallocated before capture") from error

    def bound_step(self, metadata):
        """Attach stable device routes to the exact metadata used by a layer."""
        _require(self.step is not None, "No runner step is bound")
        metadata.recent128_state = self
        metadata.recent128_step = self.step
        metadata.recent128_token_banks = self.step.token_ring_slots
        metadata.recent128_token_positions = self.step.token_positions
        metadata.recent128_token_epochs = self.step.token_request_epochs
        metadata.recent128_token_valid = self.step.token_valid
        metadata.recent128_request_banks = self.request_banks
        metadata.recent128_request_epochs = self.request_epochs
        return self.step

    attach_metadata = bound_step

    def begin_speculation(self, request_id, *, committed_end, draft_count):
        lease = self.requests[request_id]
        _positive("committed_end", committed_end, 0)
        _positive("draft_count", draft_count)
        _require(draft_count <= self.max_query_per_request,
                 "Speculative write bound exceeds ring retention contract")
        _require(lease.speculation is None, "Resolve the previous speculation transaction first")
        lease.speculation = (committed_end, draft_count)

    def resolve_speculation(self, request_id, *, accepted_tokens):
        lease = self.requests[request_id]
        _require(lease.speculation is not None, "No speculation transaction is pending")
        base, count = lease.speculation
        _positive("accepted_tokens", accepted_tokens, 0)
        _require(accepted_tokens <= count, "Accepted speculative count exceeds proposed count")
        lease.committed_end = base + accepted_tokens
        lease.speculation = None
        # Do not clear tags with a CPU round-trip. Every read is causal and
        # exact-tagged; retries overwrite the same positions before reading.
        return lease.committed_end

    def commit_request(self, request_id, *, end):
        lease = self.requests[request_id]
        _positive("end", end, 0)
        _require(lease.speculation is None, "Resolve speculation before committing request state")
        lease.committed_end = end

    def export_tail(self, request_id, *, end, completion=None):
        """Snapshot actual BF16 rows for prefix reuse/PD, outside the hot path."""
        lease = self.requests[request_id]
        _require(lease.tail_ready, "Cannot export an unavailable BF16 tail")
        _positive("end", end, 0)
        _require(lease.speculation is None, "Export only a committed tail")
        _require(end == lease.committed_end, "Snapshot end must match scheduler-committed request state")
        t, start = self.torch, max(0, end - 128)
        slots = t.arange(start, end, dtype=t.int64, device=self.device)
        slots.remainder_(self.capacity)
        layers = {}
        for key, ring in self.layers.items():
            layers[key] = LayerTail(*(t.index_select(value[lease.bank], 0, slots)
                for value in (ring.latent, ring.rope, ring.position_tags, ring.epoch_tags)))
        return TailSnapshot(start, end, lease.epoch, layers, completion)

    def import_tail(self, request_id, snapshot, *, source, completion=None):
        """Install copied prefix/PD BF16 data before admitting a nonzero start.

        Position tags are copied, not manufactured. Invalid source epochs stay
        invalid after rebinding, so exact-tag readers reject incomplete tails.
        PD requires a receive-completion event in addition to main C4 arrival.
        """
        _require(isinstance(snapshot, TailSnapshot), "Expected an actual BF16 TailSnapshot")
        _require(source in ("prefix", "pd", "resume"), "Unknown tail import source")
        lease = self.requests[request_id]
        _positive("snapshot start", snapshot.start, 0)
        _positive("snapshot end", snapshot.end, snapshot.start)
        _positive("snapshot source epoch", snapshot.source_epoch)
        _require(snapshot.end == lease.committed_end and
                 snapshot.start <= max(0, snapshot.end - 127),
                 "Imported tail does not cover the resumed query's BF16 history")
        _require(set(snapshot.layers) == set(self.layers), "Tail import must cover every local attention layer")
        event = completion if completion is not None else snapshot.completion
        _require(source != "pd" or event is not None,
                 "PD BF16 tail requires an explicit receive-completion event")
        self._wait(event)
        t, count = self.torch, snapshot.end - snapshot.start
        _require(0 <= count <= self.capacity, "Invalid tail interval")
        lease.tail_ready = False
        slots = t.arange(snapshot.start, snapshot.end, dtype=t.int64, device=self.device)
        slots.remainder_(self.capacity)
        for key, ring in self.layers.items():
            tail = snapshot.layers[key]
            expected = ((count, 512), (count, 64), (count,), (count,))
            values = (tail.latent, tail.rope, tail.position_tags, tail.epoch_tags)
            for value, shape in zip(values, expected):
                _require(tuple(value.shape) == shape and value.device == self.device,
                         "Tail tensors must have the expected shape and target device")
            _require(tail.latent.dtype == t.bfloat16 and tail.rope.dtype == t.bfloat16 and
                     tail.position_tags.dtype == t.int64 and tail.epoch_tags.dtype == t.int64,
                     "Tail payload/tag dtypes must preserve BF16 and INT64")
            new_epochs = t.full((count,), -1, dtype=t.int64, device=self.device)
            good = t.empty((count,), dtype=t.bool, device=self.device)
            t.eq(tail.epoch_tags, snapshot.source_epoch, out=good)
            new_epochs.masked_fill_(good, lease.epoch)
            for destination, value in ((ring.latent, tail.latent), (ring.rope, tail.rope),
                                       (ring.position_tags, tail.position_tags),
                                       (ring.epoch_tags, new_epochs)):
                destination[lease.bank].index_copy_(0, slots, value)
        lease.tail_ready = True
        lease.tail_source = source
        return lease

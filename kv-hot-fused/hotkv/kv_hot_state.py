"""Recent-K cache lifecycle metadata, independent of Torch and the device.

This module plans writes and addresses; it does not allocate, copy, quantize,
or transmit KV tensors. The caller must enqueue the writes/copies described by
a plan before publishing it, on the same ordered stream as its consumers (or
with explicit events). Publishing is an acknowledgement, not a device fence.

The existing C4 cache remains the full history, including a staged proposal's
causal prefix. The caller must also publish its C4 writes before publish_stage,
and keep rejected C4 positions outside the next valid prefix. This module only
tracks the extra high-precision overlay; it never drops cold-history entries.

The committed ring is separate from bounded speculative scratch. Rejecting a
proposal therefore cannot destroy the previous committed window. A request's
``committed_end`` is an EXCLUSIVE absolute token position. The next proposal
starts exactly there. Arbitrary rollback of already committed tokens is not
supported: reconstruct/restore a matching tail instead.

Per-step planning/publication is O(number of proposed/accepted tokens), never
an O(window) scan. Initial allocation and migration snapshots may be O(window).
Window zero allocates neither ring tags nor scratch-write lists. Production
callers should bypass all hot-cache machinery for their existing W=0 path.
"""
from dataclasses import dataclass
from typing import Literal, Mapping


class HotCacheStateError(RuntimeError):
    """Operation conflicts with the request lifecycle."""


class MissingHotKVError(HotCacheStateError):
    """Required high-precision payload has not been published or was evicted."""


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _request_id(value):
    if not isinstance(value, str) or not value:
        raise ValueError("request_id must be a nonempty string")
    return value


@dataclass(frozen=True, slots=True)
class WriteLocation:
    position: int
    slot: int


@dataclass(frozen=True, slots=True)
class CopyLocation:
    position: int
    source_slot: int
    destination_slot: int


@dataclass(frozen=True, slots=True)
class StagePlan:
    request_id: str
    generation: int
    step_id: int
    start: int
    count: int
    writes: tuple[WriteLocation, ...]

    @property
    def end(self):
        return self.start + self.count


@dataclass(frozen=True, slots=True)
class CommitPlan:
    request_id: str
    generation: int
    step_id: int
    accepted_count: int
    committed_end: int
    copies: tuple[CopyLocation, ...]


@dataclass(frozen=True, slots=True)
class RestorePlan:
    request_id: str
    generation: int
    committed_end: int
    kind: Literal["seed", "transfer"]
    # For seed, the caller obtains original (pre-C4) values at these positions.
    writes: tuple[WriteLocation, ...]
    # For transfer, source_slot addresses the SOURCE request's committed ring.
    copies: tuple[CopyLocation, ...]


@dataclass(frozen=True, slots=True)
class ReadLocation:
    source: Literal["cold", "ring", "scratch", "invalid"]
    position: int
    slot: int | None = None


class RecentKVState:
    """One physical request allocation's high-precision overlay metadata.

    Ring slots are [0, window); scratch slots are [window, window+max_step).
    Only ``stage`` writes scratch. ``plan_commit`` freezes reads until
    ``complete_commit``; its copies might overwrite old ring payload, so a
    prepared commit cannot be aborted. If copying fails, discard/rebuild the
    allocation instead of pretending the old ring remains intact.

    Initializing with committed_end>0 does NOT establish high-precision data.
    Call prepare_seed()/publish_restore(), or prepare_restore()/publish_restore()
    after reconstructing/copying the genuine high-precision tail. C4 decoding
    cannot restore that tail's original precision.
    """

    SNAPSHOT_VERSION = 1

    def __init__(self, request_id, generation, window, max_step, hot_dtype="c8",
                 committed_end=0, *, latent_dim=512, rope_dim=64):
        self.request_id = _request_id(request_id)
        self.generation = _integer(generation, "generation")
        self.window = _integer(window, "window")
        self.max_step = _integer(max_step, "max_step", 1)
        self.committed_end = _integer(committed_end, "committed_end")
        self.latent_dim = _integer(latent_dim, "latent_dim", 1)
        self.rope_dim = _integer(rope_dim, "rope_dim")
        if hot_dtype not in ("c8", "bf16"):
            raise ValueError("hot_dtype must be c8 or bf16")
        if hot_dtype == "c8" and latent_dim % 128:
            raise ValueError("C8 latent_dim must be divisible by its scale block 128")
        self.hot_dtype = hot_dtype
        self._tags = [-1] * window if window else None
        self._tail_ready = not window or not committed_end
        self._stage = None
        self._stage_published = False
        self._commit = None
        self._restore = None
        self._step_id = 0
        self._released = False

    @property
    def capacity(self):
        return self.window + self.max_step if self.window else 0

    @property
    def tail_ready(self):
        return (self._tail_ready and not self._released and self._restore is None
                and self._commit is None)

    @property
    def format(self):
        return {
            "hot_dtype": self.hot_dtype,
            "latent_dim": self.latent_dim,
            "rope_dim": self.rope_dim,
            "layout": ("fp8_e4m3fn_bf16rope_fp32scale128" if self.hot_dtype == "c8"
                       else "bf16latent_bf16rope"),
        }

    def _active(self):
        if self._released:
            raise HotCacheStateError("Request allocation has been released")

    def _idle(self):
        self._active()
        if self._stage is not None or self._commit is not None or self._restore is not None:
            raise HotCacheStateError("A stage, commit or restore is already pending")

    def stage(self, count):
        """Reserve a consecutive speculative suffix without touching the ring."""
        self._idle()
        count = _integer(count, "count", 1)
        if count > self.max_step:
            raise ValueError("Proposal count exceeds bounded scratch max_step")
        if not self._tail_ready:
            raise MissingHotKVError("Publish the existing committed tail before staging")
        self._step_id += 1
        writes = (tuple(WriteLocation(self.committed_end + i, self.window + i)
                        for i in range(count)) if self.window else ())
        self._stage = StagePlan(self.request_id, self.generation, self._step_id,
                                self.committed_end, count, writes)
        self._stage_published = False
        return self._stage

    def publish_stage(self, plan):
        """Acknowledge scratch writes ordered before all subsequent readers."""
        self._active()
        if plan is not self._stage or plan is None or self._commit is not None:
            raise HotCacheStateError("Stale/foreign stage plan or commit already prepared")
        if self._stage_published:
            raise HotCacheStateError("Stage has already been published")
        self._stage_published = True

    def resolve(self, position, query_position):
        """Resolve one selected logical token, preserving per-query causality.

        This is a CPU control/reference helper, not a per-TopK NPU read loop.
        A fused device reader should consume equivalent position/range metadata.
        Missing hot payload is an error, never a silent C4 fallback.
        """
        self._active()
        if self._commit is not None or self._restore is not None:
            raise HotCacheStateError("Reads are frozen during commit/restore publication")
        if isinstance(position, bool) or not isinstance(position, int):
            raise ValueError("position must be an integer")
        query_position = _integer(query_position, "query_position")
        visible_end = (self._stage.end if self._stage is not None and self._stage_published
                       else self.committed_end)
        if query_position >= visible_end:
            raise HotCacheStateError("Query is outside the published committed/provisional prefix")
        if position < 0 or position > query_position:
            return ReadLocation("invalid", position)
        if not self.window or position < max(0, query_position - self.window + 1):
            return ReadLocation("cold", position)
        if position >= self.committed_end:
            if self._stage is None or not self._stage_published or position >= self._stage.end:
                raise MissingHotKVError("Provisional hot payload is not published")
            return ReadLocation("scratch", position, self.window + position - self._stage.start)
        slot = position % self.window
        if not self._tail_ready or self._tags[slot] != position:
            raise MissingHotKVError(f"Hot position {position} is uninitialized or has been evicted")
        return ReadLocation("ring", position, slot)

    def plan_commit(self, accepted_count):
        """Prepare copies of only the accepted suffix still needed in the ring.

        For accepted_count>window, earlier accepted entries are already cold
        at the new committed end. Omitting their copies also ensures every
        destination slot appears once, avoiding scatter collisions.
        """
        self._active()
        if self._stage is None or not self._stage_published or self._commit is not None:
            raise HotCacheStateError("Commit requires one published, uncommitted stage")
        accepted_count = _integer(accepted_count, "accepted_count")
        if accepted_count > self._stage.count:
            raise ValueError("Accepted count exceeds proposed consecutive prefix")
        start = self._stage.start
        first = max(0, accepted_count - self.window)
        copies = (tuple(CopyLocation(start + i, self.window + i,
                                     (start + i) % self.window)
                        for i in range(first, accepted_count)) if self.window else ())
        self._commit = CommitPlan(self.request_id, self.generation, self._stage.step_id,
                                  accepted_count, start + accepted_count, copies)
        return self._commit

    def complete_commit(self, plan):
        """Acknowledge all planned copies ordered before subsequent consumers."""
        self._active()
        if plan is not self._commit or plan is None:
            raise HotCacheStateError("Stale/foreign commit plan")
        for copy in plan.copies:
            self._tags[copy.destination_slot] = copy.position
        self.committed_end = plan.committed_end
        self._stage = self._commit = None
        self._stage_published = False

    def abort_step(self):
        """Discard an uncommitted proposal; committed payload remains untouched."""
        self._active()
        if self._commit is not None:
            raise HotCacheStateError("Prepared copies may be in flight; release/rebuild on copy failure")
        if self._stage is None:
            raise HotCacheStateError("No stage to abort")
        self._stage = None
        self._stage_published = False

    def prepare_seed(self, committed_end=None):
        """Plan initial tail writes from genuine pre-C4 high-precision values."""
        self._idle()
        end = self.committed_end if committed_end is None else _integer(committed_end, "committed_end")
        if self.committed_end and self._tail_ready:
            raise HotCacheStateError("Cannot replace a live committed tail; release/reuse first")
        if self.committed_end and end != self.committed_end:
            raise ValueError("Seed end must match the expected initial committed_end")
        writes = (tuple(WriteLocation(p, p % self.window)
                        for p in range(max(0, end - self.window), end)) if self.window else ())
        self._restore = RestorePlan(self.request_id, self.generation, end, "seed", writes, ())
        return self._restore

    def export_snapshot(self):
        """Export metadata only; caller must transfer/fence the named payload.

        Export requires no pending proposal. Freeze this prefix against writes
        until the corresponding ring payload has been copied by the transport.
        This function itself neither freezes an external stream nor transfers KV.
        """
        self._idle()
        if not self._tail_ready:
            raise MissingHotKVError("Cannot export an uninitialized high-precision tail")
        entries = []
        if self.window:
            for p in range(max(0, self.committed_end - self.window), self.committed_end):
                slot = p % self.window
                if self._tags[slot] != p:
                    raise MissingHotKVError(f"Cannot export missing hot position {p}")
                entries.append({"position": p, "slot": slot})
        return {
            "version": self.SNAPSHOT_VERSION,
            "metadata_only": True,
            "request_id": self.request_id,
            "generation": self.generation,
            "window": self.window,
            "ring_capacity": self.window,
            "format": self.format,
            "committed_end": self.committed_end,
            "entries": entries,
        }

    def prepare_restore(self, snapshot: Mapping):
        """Validate a PD snapshot and plan copies, without making it readable."""
        self._idle()
        if self.committed_end and self._tail_ready:
            raise HotCacheStateError("Restore requires a fresh/uninitialized request allocation")
        if not isinstance(snapshot, Mapping):
            raise ValueError("Snapshot must be a metadata mapping")
        expected_keys = {"version", "metadata_only", "request_id", "generation", "window",
                         "ring_capacity", "format", "committed_end", "entries"}
        if set(snapshot) != expected_keys:
            raise ValueError("Snapshot fields do not match version 1")
        for key in ("version", "generation", "window", "ring_capacity", "committed_end"):
            _integer(snapshot[key], key)
        if snapshot["version"] != self.SNAPSHOT_VERSION or snapshot["metadata_only"] is not True:
            raise ValueError("Unsupported snapshot version or payload claim")
        if snapshot["request_id"] != self.request_id or snapshot["generation"] != self.generation:
            raise ValueError("Snapshot request identity/generation does not match the destination")
        if snapshot["window"] != self.window or snapshot["ring_capacity"] != self.window:
            raise ValueError("Snapshot window/ring capacity does not match")
        if snapshot["format"] != self.format:
            raise ValueError("Snapshot hot-cache format does not match")
        end = snapshot["committed_end"]
        if self.committed_end and end != self.committed_end:
            raise ValueError("Snapshot does not describe the expected committed prefix")
        entries = snapshot["entries"]
        count = min(self.window, end)
        if not isinstance(entries, (list, tuple)) or len(entries) != count:
            raise ValueError("Snapshot must contain the entire committed high-precision tail")
        copies = []
        for p, entry in zip(range(end - count, end), entries):
            if not isinstance(entry, Mapping) or set(entry) != {"position", "slot"}:
                raise ValueError("Invalid snapshot entry")
            position = _integer(entry["position"], "entry.position")
            slot = _integer(entry["slot"], "entry.slot")
            if position != p or slot != p % self.window:
                raise ValueError("Snapshot positions or slot mapping are not the exact committed tail")
            copies.append(CopyLocation(p, slot, slot))
        self._restore = RestorePlan(self.request_id, self.generation, end, "transfer", (), tuple(copies))
        return self._restore

    def publish_restore(self, plan):
        """Acknowledge seed writes/PD payload copies before enabling reads."""
        self._active()
        if plan is not self._restore or plan is None:
            raise HotCacheStateError("Stale/foreign restore plan")
        for write in plan.writes:
            self._tags[write.slot] = write.position
        for copy in plan.copies:
            self._tags[copy.destination_slot] = copy.position
        self.committed_end = plan.committed_end
        self._tail_ready = True
        self._restore = None

    def release(self):
        """Invalidate metadata; external allocator owns payload/event retirement."""
        self._active()
        self._released = True
        self._tags = None
        self._stage = self._commit = self._restore = None
        self._stage_published = False
        self._tail_ready = False

    def reuse(self, request_id, generation):
        """Rebind a retired physical allocation; old tags and plans stay invalid.

        The external allocator must first wait for all old GPU/transport users.
        Metadata generations alone cannot stop an in-flight DMA from writing.
        """
        if not self._released:
            raise HotCacheStateError("Release the allocation before reusing it")
        request_id = _request_id(request_id)
        generation = _integer(generation, "generation")
        if generation <= self.generation:
            raise ValueError("A reused allocation requires a strictly newer generation")
        self.request_id, self.generation = request_id, generation
        self.committed_end = 0
        self._tags = [-1] * self.window if self.window else None
        self._tail_ready = True
        self._released = False


__all__ = ["RecentKVState", "StagePlan", "CommitPlan", "RestorePlan", "WriteLocation",
           "CopyLocation", "ReadLocation", "HotCacheStateError", "MissingHotKVError"]

"""CPU contracts for the explicitly eager, single-request accuracy experiment.

This module has no Torch/NPU dependency and never changes preprocessing,
quantization or TopK selection. It is not a serving lifecycle implementation.
"""
from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np


class ContractError(ValueError):
    """An unsupported configuration or lost original BF16 cache state."""


def _integer(name, value, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ContractError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ContractError(f"{name} must be >= {minimum}")
    return value


@dataclass(frozen=True)
class Recent128Config:
    window: int = 0
    execution: str = "accuracy_eager"
    preprocess: str = "preserve"
    max_query_chunk: int = 1024
    reader: str = "native"
    library_dir: str | None = None
    source_root: str | None = None
    diagnostics_path: str | None = None
    single_pass_kernel: str = "overlay"

    def __post_init__(self):
        window = _integer("window", self.window)
        if window not in (0, 128):
            raise ContractError("sfa_recent128.window must be 0 or 128")
        query_limit = 65536 if self.execution == "serving" else 1024
        if _integer("max_query_chunk", self.max_query_chunk, 1) > query_limit:
            raise ContractError("max_query_chunk must be <=" + str(query_limit) + " for the selected reader's static query bound")
        if self.execution not in ("accuracy_eager", "serving"):
            raise ContractError("execution must be accuracy_eager or serving")
        if self.preprocess not in ("preserve", "native_accuracy"):
            raise ContractError("preprocess must be preserve or native_accuracy; the explicit override is offline accuracy only")
        if self.reader not in ("native", "fused_eligible", "fused_required", "single_pass"):
            raise ContractError("reader must be native, fused_eligible, fused_required or single_pass")
        if self.single_pass_kernel not in ("overlay", "fast_pair", "load_tile4", "kv_split4", "batched_v4"):
            raise ContractError("Unsupported single_pass_kernel")
        if self.execution == "serving":
            if self.preprocess != "preserve" or self.reader != "single_pass" or self.single_pass_kernel != "batched_v4":
                raise ContractError("serving requires preprocess=preserve, reader=single_pass, single_pass_kernel=batched_v4")
        elif self.single_pass_kernel == "batched_v4":
            raise ContractError("batched_v4 requires execution=serving")
        for name in ("library_dir", "source_root", "diagnostics_path"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ContractError(f"{name} must be a nonempty string or None")
        if self.enabled and self.diagnostics_path is None:
            raise ContractError("Enabled recent128 requires diagnostics_path for explicit fusion coverage and native fallback reasons")
        if self.enabled and self.reader != "native" and (self.library_dir is None or self.source_root is None):
            raise ContractError("A non-native reader needs library_dir and source_root; the runtime must load and verify the real library")

    @property
    def enabled(self):
        return self.window == 128

    @property
    def capacity(self):
        # The earliest query needs W history while the whole query chunk may
        # already have been written. A W-slot ring alone would overwrite it.
        if not self.enabled:
            return 0
        return ((self.window + int(self.max_query_chunk) - 1 + 127)//128)*128


def parse_recent128_config(additional_config=None):
    """Read only additional_config['sfa_recent128']; absent means window0."""
    if additional_config is None:
        return Recent128Config()
    if not isinstance(additional_config, Mapping):
        raise ContractError("additional_config must be a mapping")
    raw = additional_config.get("sfa_recent128")
    if raw is None:
        return Recent128Config()
    if not isinstance(raw, Mapping):
        raise ContractError("additional_config.sfa_recent128 must be a mapping")
    allowed = {"window", "execution", "preprocess", "max_query_chunk", "reader", "library_dir", "source_root", "diagnostics_path", "single_pass_kernel"}
    unknown = sorted(set(raw)-allowed)
    if unknown:
        hint=" Use single_pass_kernel='kv_split4' instead of kernel; reader/source_root/library_dir are already supported." if "kernel" in unknown else ""
        raise ContractError("Unsupported recent128 options; baseline quantization cannot be changed: " + ", ".join(unknown)+hint)
    return Recent128Config(**raw)


def validate_runtime_contract(config, *, preprocess_type=None, max_num_seqs=None,
                              enforce_eager=None, prefix_caching=False,
                              pd_enabled=False, pcp_size=1, dcp_size=1,
                              speculative_enabled=False, capturing=False, native_producer_available=False):
    """Fail explicitly instead of changing the framework's selected path."""
    if not config.enabled:
        return
    if config.execution == "serving":
        name = getattr(preprocess_type, "name", str(preprocess_type)).upper().rsplit(".", 1)[-1]
        if name != "PROLOG_V3" and not (name == "NATIVE" and native_producer_available is True):
            raise ContractError("serving recent128 requires selected PROLOG_V3 or a wired NATIVE BF16 producer; actual=" + name +
                ". This checks the framework route, not a TBE path or lazy compilation; provider ABI is checked separately")
        _integer("max_num_seqs", max_num_seqs, 1)
        # Batch, Graph and MTP are handled by the serving implementation.
        # This is configuration acceptance, never evidence of device support.
        return
    name = getattr(preprocess_type, "name", str(preprocess_type)).upper().rsplit(".", 1)[-1]
    if name != "NATIVE":
        raise ContractError("recent128 accuracy_eager requires an already selected NATIVE path; "
                            "PROLOG_V3 has no original BF16 KV output here and is not silently disabled")
    if _integer("max_num_seqs", max_num_seqs, 1) != 1:
        raise ContractError("recent128 initial accuracy experiment requires max_num_seqs=1")
    if enforce_eager is not True or capturing:
        raise ContractError("recent128 execution='accuracy_eager' requires enforce_eager=True and no graph capture")
    if prefix_caching or pd_enabled or speculative_enabled:
        raise ContractError("recent128 initial contract excludes prefix caching, PD and speculative/MTP execution")
    if _integer("pcp_size", pcp_size, 1) != 1 or _integer("dcp_size", dcp_size, 1) != 1:
        raise ContractError("recent128 initial contract excludes PCP and DCP")


@dataclass(frozen=True)
class StepPlan:
    enabled: bool
    start: int | None
    end: int | None
    reset: bool
    positions: np.ndarray | None
    capacity: int


def plan_step(config, *, seq_len, query_count, previous_end):
    """Plan one contiguous request chunk, preserving ring history across chunks.

    seq_len is the exclusive end after this chunk; previous_end is the last
    successfully committed end. Callers commit previous_end only after success.
    Start0 begins a new request and requires clearing all per-layer ring tags.
    A disabled plan does not inspect arguments or allocate position arrays.
    """
    if not config.enabled:
        return StepPlan(False, None, None, False, None, 0)
    end = _integer("seq_len", seq_len)
    count = _integer("query_count", query_count, 1)
    if count > config.max_query_chunk:
        raise ContractError("query_count exceeds configured max_query_chunk; refusing ring overwrite")
    start = end-count
    if start < 0:
        raise ContractError("seq_len must include all current query tokens")
    reset = start == 0
    if not reset:
        if previous_end is None or start != _integer("previous_end", previous_end):
            raise ContractError("Noncontiguous request: prefix reuse, interleaving, skipped chunks or speculative rollback is unsupported")
    return StepPlan(True, start, end, reset, np.arange(start, end, dtype=np.int64), config.capacity)


def _integer_array(name, value):
    array = np.asarray(value)
    if array.dtype.kind not in "iu":
        raise ContractError(f"{name} must contain integer values without a float conversion")
    if array.dtype.kind == "u" and array.size and array.max() > np.iinfo(np.int64).max:
        raise ContractError(f"{name} exceeds signed int64 range")
    return array.astype(np.int64, copy=False)


def partition_reference(topk, *, positions, tags=None, window=128, capacity=None):
    """Stable partition of unchanged causal TopK, with exact original-KV tags.

    Indices may be [T,S] or [T,1,S]. Returned indices are always [T,1,width],
    padded with -1; true zero counts never manufacture a dummy token. The
    runtime must invoke native BF16 attention at each row/group's compact count.
    Future IDs are causally masked, never promoted to hot/cold attention.
    A selected window token without its exact original-BF16 tag is an error,
    not permission to dequantize C4 or silently send it to the cold reader.
    """
    window = _integer("window", window)
    if window not in (0, 128):
        raise ContractError("window must be 0 or 128")
    pos = _integer_array("positions", positions)
    if pos.ndim != 1 or np.any(pos < 0):
        raise ContractError("positions must be a nonnegative 1D logical-token array")
    indices = _integer_array("topk", topk)
    if indices.ndim == 2:
        indices = indices[:, None, :]
    if indices.ndim != 3 or indices.shape[:2] != (len(pos), 1):
        raise ContractError("topk must have shape [T,S] or [T,1,S] matching positions")
    if np.any(indices < -1):
        raise ContractError("Only -1 is a valid TopK padding sentinel")
    width = indices.shape[-1]
    hot_width = min(window, width)
    old = np.full(indices.shape, -1, dtype=np.int64)
    recent_global = np.full((len(pos), 1, hot_width), -1, dtype=np.int64)
    recent_ring = np.full_like(recent_global, -1)
    old_counts, recent_counts = np.zeros(len(pos), np.int64), np.zeros(len(pos), np.int64)
    ring_tags = None
    if window:
        if tags is not None:
            ring_tags = _integer_array("tags", tags)
            if ring_tags.ndim != 1:
                raise ContractError("tags must be a 1D ring of original BF16 logical positions")
        if capacity is None:
            capacity = 0 if ring_tags is None else len(ring_tags)
        capacity = _integer("capacity", capacity, 1)
        if ring_tags is not None and len(ring_tags) != capacity:
            raise ContractError("tags length must equal ring capacity")
    for row, p in enumerate(pos):
        raw = indices[row, 0]
        padded = raw == -1
        if padded.any() and np.any(raw[np.argmax(padded):] >= 0):
            raise ContractError("-1 padding must be a trailing suffix")
        selected = raw[(raw >= 0) & (raw <= p)]
        if len(np.unique(selected)) != len(selected):
            raise ContractError("Duplicate causal TopK indices would change attention mass")
        hot_mask = selected >= p-window+1 if window else np.zeros(len(selected), dtype=bool)
        cold_ids, hot_ids = selected[~hot_mask], selected[hot_mask]
        old_counts[row], recent_counts[row] = len(cold_ids), len(hot_ids)
        old[row, 0, :len(cold_ids)] = cold_ids
        if len(hot_ids):
            slots = hot_ids % capacity
            if ring_tags is None or not np.array_equal(ring_tags[slots], hot_ids):
                raise ContractError("Selected recent token has no exact original BF16 tag in the ring")
            recent_global[row, 0, :len(hot_ids)] = hot_ids
            recent_ring[row, 0, :len(hot_ids)] = slots
    return dict(positions=pos.copy(), old=old, recent_global=recent_global, recent_ring=recent_ring,
                old_counts=old_counts, recent_counts=recent_counts,
                old_present=old_counts > 0, recent_present=recent_counts > 0)


@dataclass(frozen=True)
class FusionDecision:
    use_fused: bool
    reason: str


def fusion_eligibility(config, *, hot_count, heads, latent_dim, rope_dim,
                       query_count, library_available):
    """Describe direct-ABI eligibility after explicit accuracy-only adapters.

    Pass the H/R actually sent to the direct reader (including any reported
    padding adapter); latent D must remain the model's actual D512. Both mixed
    readers use native compact calls for ineligible groups. fused_required
    requires a real loaded library and fusion for every eligible group; it does
    not invent hot tokens to turn counts0..127 into count128. Callers must report
    row counts/reasons and mark fusion NOT_EXERCISED if fused_rows stays zero.

    single_pass is a separate whole-TopK operator/runtime, not the old ABI4 hot
    merge. It never uses this helper's count128 gate. The dedicated reader
    validates its actual B1/R64 contract and explicit local-head padding before
    calling the new ABI; those accuracy-only adapters confer no serving claim.
    """
    if not config.enabled:
        return FusionDecision(False, "window_disabled")
    if config.reader == "native":
        return FusionDecision(False, "reader_native")
    if config.reader == "single_pass":
        return FusionDecision(False, "single_pass_whole_topk_runtime")
    if not library_available:
        if config.reader == "fused_required":
            raise ContractError("fused_required could not load the real direct library")
        return FusionDecision(False, "library_unavailable")
    checks = ((hot_count == 128, "hot_count_not_128"),
              (heads == 64, "heads_not_64"),
              (latent_dim == 512, "latent_dim_not_512"),
              (rope_dim == 64, "rope_dim_not_64"),
              (isinstance(query_count, (int, np.integer)) and not isinstance(query_count, bool)
               and 1 <= query_count <= 1024, "query_count_out_of_range"))
    for passed, reason in checks:
        if not passed:
            return FusionDecision(False, reason)
    return FusionDecision(True, "eligible_exact_compact128_direct_contract")

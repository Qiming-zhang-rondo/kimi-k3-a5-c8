"""Experimental single-launch stable partition for native attention indices.

Each input row is a unique TopK selection in original selection order. Visible
indices satisfy 0 <= index <= query_position; -1 and future entries are ignored.
Recent keys satisfy query_position-window < index <= query_position. Cold and
recent outputs retain their input order and have -1 suffixes. An empty output
starts with dummy index 0 for native readers; its presence flag is false.

Only index preparation happens here: no attention, cache decoding or merge.
The kernel uses integer prefix scans, not sorting/nonzero. Its valid-prefix
scatter, suffix padding and empty-row dummy stores write disjoint locations,
so there is no fill-then-scatter dependency or missing inter-store barrier.

Inputs must be unique per row. A recent capacity >= min(window, TopK capacity)
then cannot overflow. An explicit invalid_rows output flags capacity overflow,
unsupported negatives and future indices; consumers must reject such rows,
not interpret filtered/truncated results. Bounds masks also prevent writes
outside allocated output buffers. Duplicate detection remains caller-side.

Official Ascend docs list forward int32 cumsum support on Ascend 950:
https://triton-ascend.readthedocs.io/en/latest/python-api/generated/triton.language.cumsum.html
Compilation and performance on the installed A5 Triton still need validation.

The default two_scan variant is unchanged. Optional packed_one_scan encodes
visible/recent counts in separate integer bit fields and uses one prefix scan.
It rejects shapes whose worst-case packed prefix could overflow signed int32;
there is no implicit fallback to another implementation.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PartitionLayout:
    rows: int
    sparse_capacity: int
    hot_width: int
    window: int
    ring_capacity: int
    remap_ring: bool
    block: int
    variant: str
    packed_shift: int


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_inputs(indices, positions, *, window, ring_capacity, hot_width=None, remap_ring=True,
                    variant="two_scan"):
    """Inspect only host metadata, never tensor contents or device values."""
    window = _positive_int(window, "window")
    ring_capacity = _positive_int(ring_capacity, "ring_capacity")
    if not isinstance(remap_ring, bool):
        raise ValueError("remap_ring must be a bool")
    if variant not in ("two_scan", "packed_one_scan"):
        raise ValueError("variant must be two_scan or packed_one_scan")
    shape = tuple(indices.shape)
    if len(shape) != 3 or shape[0] < 1 or shape[1] != 1 or shape[2] < 1:
        raise ValueError("indices must have shape [T,1,S] with T,S > 0")
    rows, _, capacity = shape
    if str(indices.dtype) != "torch.int32":
        raise ValueError("indices must have dtype torch.int32")
    if tuple(positions.shape) != (rows,):
        raise ValueError("positions must have shape [T]")
    if str(positions.dtype) not in ("torch.int32", "torch.int64"):
        raise ValueError("positions must have dtype torch.int32 or torch.int64")
    device = str(indices.device)
    if device.split(":", 1)[0] not in ("npu", "privateuseone"):
        raise ValueError("PreparedPartition requires Ascend NPU tensors")
    for name, tensor in (("indices", indices), ("positions", positions)):
        if str(tensor.device) != device:
            raise ValueError(f"{name} must be on the same NPU as indices")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous; prepare outside timing")
    if remap_ring and ring_capacity < window:
        raise ValueError("ring_capacity must be >= window when remapping recent keys")
    minimum_width = min(window, capacity)
    hot_width = minimum_width if hot_width is None else _positive_int(hot_width, "hot_width")
    if not minimum_width <= hot_width <= capacity:
        raise ValueError("hot_width must be between min(window,S) and S")
    # BASE strictly exceeds every possible visible prefix. Consequently the
    # low field cannot carry into the recent field, even for an all-hot row.
    packed_shift = capacity.bit_length()
    if variant == "packed_one_scan" and capacity * ((1 << packed_shift) + 1) > 2147483647:
        raise ValueError("packed_one_scan worst-case prefix exceeds signed int32 capacity")
    return PartitionLayout(rows, capacity, hot_width, window, ring_capacity, remap_ring,
                           1 << (capacity - 1).bit_length(), variant, packed_shift)


_KERNEL = None


def _load_kernel():
    global _KERNEL, tl
    if _KERNEL is not None:
        return _KERNEL
    try:
        import triton
        import triton.language as tl
    except ImportError as exc:
        raise RuntimeError("Stable partition needs the Triton build matching this Ascend environment") from exc

    @triton.jit
    def _partition_kernel(INDICES, POSITIONS, OLD, RECENT_GLOBAL, RECENT_RING,
                           OLD_PRESENT, RECENT_PRESENT, INVALID_ROWS,
                           S: tl.constexpr, HW: tl.constexpr, WINDOW: tl.constexpr,
                           CAP: tl.constexpr, REMAP_RING: tl.constexpr, BLOCK: tl.constexpr,
                           PACKED_SCAN: tl.constexpr, PACKED_SHIFT: tl.constexpr):
        row = tl.program_id(0)
        lane = tl.arange(0, BLOCK)
        key = tl.load(INDICES + row.to(tl.int64) * S + lane, mask=lane < S, other=-1)
        position = tl.load(POSITIONS + row).to(tl.int64)
        visible = (lane < S) & (key >= 0) & (key.to(tl.int64) <= position)
        recent = visible & (key.to(tl.int64) > position - WINDOW)
        old = visible & ~recent
        if PACKED_SCAN:
            encoded = visible.to(tl.int32) + (recent.to(tl.int32) << PACKED_SHIFT)
            prefix = tl.cumsum(encoded, 0)
            recent_prefix = prefix >> PACKED_SHIFT
            visible_prefix = prefix & ((1 << PACKED_SHIFT) - 1)
            old_rank = visible_prefix - recent_prefix - 1
            recent_rank = recent_prefix - 1
            packed_count = tl.max(prefix, 0)
            recent_count = packed_count >> PACKED_SHIFT
            old_count = (packed_count & ((1 << PACKED_SHIFT) - 1)) - recent_count
        else:
            old_rank = tl.cumsum(old.to(tl.int32), 0) - 1
            recent_rank = tl.cumsum(recent.to(tl.int32), 0) - 1
            old_count = tl.sum(old.to(tl.int32), 0)
            recent_count = tl.sum(recent.to(tl.int32), 0)
        tl.store(OLD_PRESENT + row, old_count > 0)
        tl.store(RECENT_PRESENT + row, recent_count > 0)
        bad_key = (lane < S) & ((key < -1) | ((key >= 0) & (key.to(tl.int64) > position)))
        invalid_row = (position < 0) | (recent_count > HW) | (tl.sum(bad_key.to(tl.int32), 0) > 0)
        tl.store(INVALID_ROWS + row, invalid_row)

        old_base = row.to(tl.int64) * S
        # Prefix [0,count) and suffix [max(count,1),S) are disjoint. The dummy
        # store covers only element 0 of empty rows, disjoint from both.
        tl.store(OLD + old_base + old_rank, key, mask=old)
        tl.store(OLD + old_base + lane, -1,
                 mask=(lane < S) & (lane >= tl.maximum(old_count, 1)))
        tl.store(OLD + old_base, 0, mask=old_count == 0)

        recent_base = row.to(tl.int64) * HW
        hot_store = recent & (recent_rank < HW)
        hot_padding = (lane < HW) & (lane >= tl.maximum(recent_count, 1))
        tl.store(RECENT_GLOBAL + recent_base + recent_rank, key, mask=hot_store)
        tl.store(RECENT_GLOBAL + recent_base + lane, -1, mask=hot_padding)
        tl.store(RECENT_GLOBAL + recent_base, 0, mask=recent_count == 0)
        if REMAP_RING:
            tl.store(RECENT_RING + recent_base + recent_rank, key % CAP, mask=hot_store)
            tl.store(RECENT_RING + recent_base + lane, -1, mask=hot_padding)
            tl.store(RECENT_RING + recent_base, 0, mask=recent_count == 0)

    _KERNEL = _partition_kernel
    return _KERNEL


class PreparedPartition:
    """Allocate fixed outputs once, then enqueue one partition kernel per call.

    Update input tensors in place; inputs and returned tensors must remain alive
    through stream completion. Warm up before graph capture. ``call()`` returns
    the same dict/buffers every time and does not allocate or synchronize.
    ``invalid_rows`` is a boolean [T] diagnostic and must be checked outside
    timing for unvalidated inputs. Uniqueness is a caller precondition, not checked by
    this scan kernel. ``recent_ring`` aliases ``recent_global`` for split-C4
    controls (remap_ring=False), avoiding a redundant allocation and store.
    """

    def __init__(self, indices, positions, *, window, ring_capacity, hot_width=None, remap_ring=True,
                 variant="two_scan"):
        self.layout = validate_inputs(indices, positions, window=window, ring_capacity=ring_capacity,
                                      hot_width=hot_width, remap_ring=remap_ring, variant=variant)
        import torch
        layout = self.layout
        device = indices.device
        old = torch.empty((layout.rows, 1, layout.sparse_capacity), dtype=torch.int32, device=device)
        recent_global = torch.empty((layout.rows, 1, layout.hot_width), dtype=torch.int32, device=device)
        recent_ring = (torch.empty_like(recent_global) if remap_ring else recent_global)
        old_present = torch.empty((layout.rows,), dtype=torch.bool, device=device)
        recent_present = torch.empty((layout.rows,), dtype=torch.bool, device=device)
        invalid_rows = torch.empty((layout.rows,), dtype=torch.bool, device=device)
        self.outputs = dict(old=old, recent_global=recent_global, recent_ring=recent_ring,
                            old_present=old_present, recent_present=recent_present, invalid_rows=invalid_rows)
        self._inputs = (indices, positions, old, recent_global, recent_ring,
                        old_present, recent_present, invalid_rows)
        self._kernel = _load_kernel()
        self._grid = (layout.rows,)
        self._constants = dict(S=layout.sparse_capacity, HW=layout.hot_width, WINDOW=layout.window,
                               CAP=layout.ring_capacity, REMAP_RING=layout.remap_ring, BLOCK=layout.block,
                               PACKED_SCAN=variant == "packed_one_scan", PACKED_SHIFT=layout.packed_shift)
        implementation = ("ascend_triton_packed_cumsum_partition" if variant == "packed_one_scan"
                          else "ascend_triton_cumsum_partition")
        self.metadata = dict(implementation=implementation, kernel_launches=1,
                             attention_launches=0, stable=True, host_synchronization=False,
                             variant=variant, source_scans_per_row=1 if variant == "packed_one_scan" else 2,
                             packed_prefix_bound=(layout.sparse_capacity * ((1 << layout.packed_shift) + 1)
                                                  if variant == "packed_one_scan" else None),
                             output_allocation="constructor_only", sparse_capacity=layout.sparse_capacity,
                             hot_width=layout.hot_width, block=layout.block, remap_ring=remap_ring,
                             invalid_rows_output=True, input_contract="unique visible TopK; -1 padding",
                             runtime_validation="capacity/negative/future checks; uniqueness is caller-validated",
                             status="EXPERIMENTAL_REQUIRES_A5_VALIDATION")

    def call(self):
        self._kernel[self._grid](*self._inputs, **self._constants)
        return self.outputs

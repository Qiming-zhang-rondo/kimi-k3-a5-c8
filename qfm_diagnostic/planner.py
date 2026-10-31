"""CPU-only metadata prototype for batched QuantFlashMLA virtual sequences.

One owner denotes one original request and its existing physical KV block table.
Splitting an owner's query rows never copies its KV data.  This module proves
metadata arithmetic only; it does not establish native operator, graph, cache
layout, or distributed-communication compatibility.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class QFMPlan:
    """Metadata for contiguous query segments in the original packed Q buffer.

    ``q_start``/``q_end`` are absolute half-open offsets, including any trailing
    graph-padding rows.  ``cu_seqlens_q`` is their cumulative physical extent;
    ``seqused_q`` distinguishes active rows from padding.  ``owners`` indexes
    the caller's original request/block-table rows, or is -1 for padding.
    ``cache_seqlens`` counts keys visible to each virtual sequence, globally
    for a single rank or locally for the configured DCP rank.

    Mask 3 means right-aligned causal attention.  Mask 0 is used for one-query
    DCP segments whose rank-local key prefix is already exact.  A DCP segment
    can have an empty local prefix: a native integration must handle its
    neutral attention/LSE contribution explicitly if the operator cannot.
    """

    owners: tuple[int, ...]
    q_start: tuple[int, ...]
    q_end: tuple[int, ...]
    cu_seqlens_q: tuple[int, ...]
    seqused_q: tuple[int, ...]
    cache_seqlens: tuple[int, ...]
    mask_mode: int


def _integer(name: str, value: int, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not {type(value).__name__}")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _topology(dcp_size: int, dcp_rank: int, interleave: int) -> tuple[int, int, int]:
    dcp_size = _integer("dcp_size", dcp_size, 1)
    dcp_rank = _integer("dcp_rank", dcp_rank)
    interleave = _integer("interleave", interleave, 1)
    if dcp_rank >= dcp_size:
        raise ValueError("dcp_rank must be less than dcp_size")
    return dcp_size, dcp_rank, interleave


def local_prefix_length(
    global_prefix: int, dcp_size: int, dcp_rank: int, interleave: int = 1
) -> int:
    """Count a rank's keys in global positions ``[0, global_prefix)``.

    Ownership is ``(position // interleave) % dcp_size``; each rank's physical
    cache is assumed to retain its owned keys in increasing global-position
    order.  This is a configurable mathematical ownership contract, not a
    claim about the current vLLM-Ascend cache placement or HCCL behavior.
    """
    global_prefix = _integer("global_prefix", global_prefix)
    dcp_size, dcp_rank, interleave = _topology(dcp_size, dcp_rank, interleave)
    cycles, remainder = divmod(global_prefix, dcp_size * interleave)
    return cycles * interleave + min(
        max(remainder - dcp_rank * interleave, 0), interleave
    )


def plan_qfm_batch(
    query_lengths: Sequence[int],
    cache_lengths: Sequence[int],
    *,
    max_query_rows: int = 16,
    padded_tokens: int | None = None,
    dcp_size: int = 1,
    dcp_rank: int = 0,
    interleave: int = 1,
) -> QFMPlan:
    """Plan a batch without allocating, duplicating, or modifying KV storage.

    Inputs describe original requests in packed-Q order.  ``cache_lengths``
    are GLOBAL logical lengths *after* writing this step's query/KV rows;
    the preceding cached/APC prefix is therefore ``cache_length-query_length``.
    A prefix need not be block-aligned.  For non-DCP, a segment ending at
    request-relative row ``end`` sees ``prefix+end`` keys and uses mask 3.

    DCP uses one active query per virtual sequence and mask 0, with that
    query's exact rank-local prefix count.  No collectives or attention/LSE
    reduction are implemented here.  ``padded_tokens`` is the total physical
    Q-row capacity, not a number of extra rows.  Padding segments remain at
    most ``max_query_rows`` rows even though their used-query count is zero.

    An empty batch without padding returns an empty plan (no operator call).
    Native per-call sequence limits and graph buffer capacities are deliberately
    left to a later integration; the only operator shape bound modeled here
    is at most 16 physical query rows per virtual sequence.
    """
    max_query_rows = _integer("max_query_rows", max_query_rows, 1)
    if max_query_rows > 16:
        raise ValueError("max_query_rows must be at most 16")
    dcp_size, dcp_rank, interleave = _topology(dcp_size, dcp_rank, interleave)
    queries = tuple(
        _integer(f"query_lengths[{i}]", n) for i, n in enumerate(query_lengths)
    )
    caches = tuple(
        _integer(f"cache_lengths[{i}]", n) for i, n in enumerate(cache_lengths)
    )
    if len(queries) != len(caches):
        raise ValueError("query_lengths and cache_lengths must have the same length")
    if any(cache < query for query, cache in zip(queries, caches)):
        raise ValueError("each cache length must include all of its query rows")
    total_queries = sum(queries)
    padded_tokens = (
        total_queries
        if padded_tokens is None
        else _integer("padded_tokens", padded_tokens)
    )
    if padded_tokens < total_queries:
        raise ValueError("padded_tokens must cover all query rows")

    owners: list[int] = []
    starts: list[int] = []
    ends: list[int] = []
    used: list[int] = []
    visible: list[int] = []
    offset = 0
    segment_limit = 1 if dcp_size > 1 else max_query_rows
    for owner, (query_length, cache_length) in enumerate(zip(queries, caches)):
        prefix = cache_length - query_length
        for start in range(0, query_length, segment_limit):
            end = min(start + segment_limit, query_length)
            owners.append(owner)
            starts.append(offset + start)
            ends.append(offset + end)
            used.append(end - start)
            global_visible = prefix + end
            visible.append(
                local_prefix_length(global_visible, dcp_size, dcp_rank, interleave)
                if dcp_size > 1
                else global_visible
            )
        offset += query_length

    while offset < padded_tokens:
        end = min(offset + max_query_rows, padded_tokens)
        owners.append(-1)
        starts.append(offset)
        ends.append(end)
        used.append(0)
        visible.append(0)
        offset = end

    return QFMPlan(
        owners=tuple(owners),
        q_start=tuple(starts),
        q_end=tuple(ends),
        cu_seqlens_q=(0, *ends),
        seqused_q=tuple(used),
        cache_seqlens=tuple(visible),
        mask_mode=0 if dcp_size > 1 else 3,
    )

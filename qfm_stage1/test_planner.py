"""Independent causal-attention arithmetic checks; no NPU/HCCL claims."""

import math
from dataclasses import FrozenInstanceError

import pytest
from planner import local_prefix_length, plan_qfm_batch


def _offsets(lengths):
    result = [0]
    for length in lengths:
        result.append(result[-1] + length)
    return result


def _assert_plan_invariants(plan, query_lengths, padded_tokens=None):
    total_queries = sum(query_lengths)
    physical_rows = total_queries if padded_tokens is None else padded_tokens
    count = len(plan.owners)
    assert all(
        len(field) == count
        for field in (plan.q_start, plan.q_end, plan.seqused_q, plan.cache_seqlens)
    )
    assert plan.cu_seqlens_q == (0, *plan.q_end)
    assert plan.q_start == plan.cu_seqlens_q[:-1]
    assert plan.cu_seqlens_q[-1] == physical_rows
    assert sum(plan.seqused_q) == total_queries
    offsets = _offsets(query_lengths)
    actual_rows = []
    for owner, start, end, used, cachelen in zip(
        plan.owners, plan.q_start, plan.q_end, plan.seqused_q, plan.cache_seqlens
    ):
        assert 1 <= end - start <= 16
        assert cachelen >= 0
        if owner == -1:
            assert used == cachelen == 0
            assert start >= total_queries
        else:
            assert 0 <= owner < len(query_lengths)
            assert offsets[owner] <= start < end <= offsets[owner + 1]
            assert used == end - start
            actual_rows.extend(range(start, end))
    # Every original query occurs once, with no reordering or physical Q copies.
    assert actual_rows == list(range(total_queries))


def _query(owner, row):
    return (math.sin(row + owner * 0.7), math.cos(row * 0.3 - owner))


def _key(owner, position):
    return (math.cos(position * 0.17 + owner), math.sin(position * 0.23))


def _value(owner, position):
    return (math.sin(position * 0.11 + owner), math.cos(position * 0.19 - owner))


def _attention(query, owner, key_positions):
    """Dense reference attention and logsumexp over explicit global keys."""
    positions = list(key_positions)
    if not positions:
        return (0.0, 0.0), -math.inf
    scores = [sum(a * b for a, b in zip(query, _key(owner, k))) for k in positions]
    maximum = max(scores)
    weights = [math.exp(score - maximum) for score in scores]
    denominator = sum(weights)
    output = tuple(
        sum(weight * _value(owner, k)[axis] for weight, k in zip(weights, positions))
        / denominator
        for axis in range(2)
    )
    return output, maximum + math.log(denominator)


@pytest.mark.parametrize("segment_limit", [1, 3, 7, 16])
@pytest.mark.parametrize(
    "queries,caches",
    [
        ([1, 17, 0, 33], [5, 24, 0, 70]),
        ([18, 5, 32], [18, 134, 161]),  # first prefill and partial APC prefixes
        ([0, 1, 0], [321, 130, 50]),
    ],
)
def test_segmented_causal_mask_and_dense_attention(queries, caches, segment_limit):
    plan = plan_qfm_batch(queries, caches, max_query_rows=segment_limit)
    _assert_plan_invariants(plan, queries)
    assert plan.mask_mode == 3
    offsets = _offsets(queries)
    for owner, start, end, cachelen in zip(
        plan.owners, plan.q_start, plan.q_end, plan.cache_seqlens
    ):
        assert end - start <= segment_limit
        prefix = caches[owner] - queries[owner]
        for packed_row in range(start, end):
            request_row = packed_row - offsets[owner]
            # Interpret the native right-aligned triangle independently of
            # the planner's P+segment_end construction.
            virtual_row = packed_row - start
            mask = [
                key <= cachelen - (end - start) + virtual_row for key in range(cachelen)
            ]
            actual_keys = [key for key, allowed in enumerate(mask) if allowed]
            expected_keys = list(range(prefix + request_row + 1))
            assert actual_keys == expected_keys
            query = _query(owner, request_row)
            actual, actual_lse = _attention(query, owner, actual_keys)
            expected, expected_lse = _attention(query, owner, expected_keys)
            assert actual == pytest.approx(expected, abs=1e-14)
            assert actual_lse == pytest.approx(expected_lse, abs=1e-14)


def test_8192_query_chunk_131072_context_reuses_one_cache_owner():
    plan = plan_qfm_batch([8192], [131072])
    _assert_plan_invariants(plan, [8192])
    assert len(plan.owners) == 512
    assert set(plan.owners) == {0}
    assert plan.seqused_q == (16,) * 512
    assert plan.cache_seqlens[0] == 122896
    assert plan.cache_seqlens[-1] == 131072
    # Owner lookup repeats the SAME original block-table reference, not KV.
    original_block_tables = [object()]
    assert all(
        original_block_tables[owner] is original_block_tables[0]
        for owner in plan.owners
    )
    assert not hasattr(plan, "kv_cache")
    for start, end, cachelen in zip(plan.q_start, plan.q_end, plan.cache_seqlens):
        for row in (start, end - 1):
            last_allowed_key = cachelen - (end - start) + (row - start)
            assert last_allowed_key == 122880 + row


@pytest.mark.parametrize("dcp_size", [1, 2, 3, 8])
@pytest.mark.parametrize("interleave", [1, 4, 16])
def test_local_prefix_matches_independent_position_ownership(dcp_size, interleave):
    for prefix in range(4 * dcp_size * interleave + 3):
        counts = []
        for rank in range(dcp_size):
            expected = sum(
                (position // interleave) % dcp_size == rank
                for position in range(prefix)
            )
            actual = local_prefix_length(prefix, dcp_size, rank, interleave)
            assert actual == expected
            counts.append(actual)
        assert sum(counts) == prefix


@pytest.mark.parametrize("dcp_size,interleave", [(2, 1), (3, 4), (8, 16)])
def test_dcp_one_query_prefix_and_lse_merge_equal_dense(dcp_size, interleave):
    queries, caches = [5, 19, 2], [5, 148, 33]
    offsets = _offsets(queries)
    plans = [
        plan_qfm_batch(
            queries, caches, dcp_size=dcp_size, dcp_rank=rank, interleave=interleave
        )
        for rank in range(dcp_size)
    ]
    for plan in plans:
        _assert_plan_invariants(plan, queries)
        assert plan.mask_mode == 0
        assert set(plan.seqused_q) == {1}
    saw_empty_shard = False
    for owner, query_length in enumerate(queries):
        # Build the physical rank caches by enumerating ownership, not by the
        # prefix-length formula used by the implementation.
        rank_cache_positions = [
            [k for k in range(caches[owner]) if (k // interleave) % dcp_size == rank]
            for rank in range(dcp_size)
        ]
        for request_row in range(query_length):
            packed_row = offsets[owner] + request_row
            prefix = caches[owner] - query_length + request_row + 1
            query = _query(owner, request_row)
            partials = []
            combined_keys = []
            for rank, plan in enumerate(plans):
                assert plan.owners[packed_row] == owner
                assert plan.q_start[packed_row] == packed_row
                assert plan.q_end[packed_row] == packed_row + 1
                count = plan.cache_seqlens[packed_row]
                # Mask 0: all selected local keys are visible to this query.
                keys = rank_cache_positions[rank][:count]
                assert keys == [
                    k for k in range(prefix) if (k // interleave) % dcp_size == rank
                ]
                combined_keys.extend(keys)
                saw_empty_shard |= not keys
                partials.append(_attention(query, owner, keys))
            assert sorted(combined_keys) == list(range(prefix))
            assert len(set(combined_keys)) == len(combined_keys)
            maximum_lse = max(lse for _, lse in partials)
            rank_weights = [math.exp(lse - maximum_lse) for _, lse in partials]
            merged = tuple(
                sum(
                    weight * output[axis]
                    for weight, (output, _) in zip(rank_weights, partials)
                )
                / sum(rank_weights)
                for axis in range(2)
            )
            merged_lse = maximum_lse + math.log(sum(rank_weights))
            expected, expected_lse = _attention(query, owner, range(prefix))
            assert merged == pytest.approx(expected, abs=2e-14)
            assert merged_lse == pytest.approx(expected_lse, abs=2e-14)
    assert saw_empty_shard


@pytest.mark.parametrize("dcp_size,dcp_rank", [(1, 0), (4, 0), (4, 3)])
@pytest.mark.parametrize("padded_tokens", [17, 18, 32, 80])
def test_graph_padding_has_bounded_physical_segments(dcp_size, dcp_rank, padded_tokens):
    plan = plan_qfm_batch(
        [17], [146], padded_tokens=padded_tokens, dcp_size=dcp_size, dcp_rank=dcp_rank
    )
    _assert_plan_invariants(plan, [17], padded_tokens)
    dummy_lengths = [
        end - start
        for owner, start, end in zip(plan.owners, plan.q_start, plan.q_end)
        if owner == -1
    ]
    assert sum(dummy_lengths) == padded_tokens - 17
    assert all(length <= 16 for length in dummy_lengths)


def test_padding_respects_smaller_configured_limit():
    plan = plan_qfm_batch([5], [5], max_query_rows=3, padded_tokens=17)
    _assert_plan_invariants(plan, [5], 17)
    assert max(end - start for start, end in zip(plan.q_start, plan.q_end)) == 3


@pytest.mark.parametrize("queries,caches", [([], []), ([0], [0]), ([0, 0], [130, 41])])
def test_empty_active_batches(queries, caches):
    plan = plan_qfm_batch(queries, caches)
    _assert_plan_invariants(plan, queries)
    assert plan.owners == ()
    assert plan.cu_seqlens_q == (0,)
    padded = plan_qfm_batch(queries, caches, padded_tokens=33)
    _assert_plan_invariants(padded, queries, 33)
    assert padded.owners == (-1, -1, -1)
    assert padded.seqused_q == padded.cache_seqlens == (0, 0, 0)


def test_input_storage_is_not_mutated_and_plan_is_immutable():
    queries, caches = [17, 1], [146, 22]
    plan = plan_qfm_batch(queries, caches)
    assert queries == [17, 1]
    assert caches == [146, 22]
    queries[0] = 0
    caches[0] = 0
    assert plan.seqused_q == (16, 1, 1)
    assert plan.cache_seqlens == (145, 146, 22)
    with pytest.raises(FrozenInstanceError):
        plan.mask_mode = 0


@pytest.mark.parametrize(
    "queries,caches,kwargs,error",
    [
        ([1], [], {}, ValueError),
        ([-1], [0], {}, ValueError),
        ([1], [-1], {}, ValueError),
        ([2], [1], {}, ValueError),
        ([True], [1], {}, TypeError),
        ([1.0], [1], {}, TypeError),
        ([1], ["1"], {}, TypeError),
        ([1], [1], {"max_query_rows": 0}, ValueError),
        ([1], [1], {"max_query_rows": 17}, ValueError),
        ([1], [1], {"max_query_rows": 1.5}, TypeError),
        ([1], [1], {"padded_tokens": 0}, ValueError),
        ([1], [1], {"padded_tokens": -1}, ValueError),
        ([1], [1], {"padded_tokens": True}, TypeError),
        ([1], [1], {"dcp_size": 0}, ValueError),
        ([1], [1], {"dcp_size": 2, "dcp_rank": 2}, ValueError),
        ([1], [1], {"dcp_rank": -1}, ValueError),
        ([1], [1], {"dcp_rank": 0.0}, TypeError),
        ([1], [1], {"interleave": 0}, ValueError),
    ],
)
def test_invalid_batch_inputs(queries, caches, kwargs, error):
    with pytest.raises(error):
        plan_qfm_batch(queries, caches, **kwargs)


@pytest.mark.parametrize(
    "prefix,error", [(-1, ValueError), (1.5, TypeError), (True, TypeError)]
)
def test_invalid_prefix_inputs(prefix, error):
    with pytest.raises(error):
        local_prefix_length(prefix, 2, 0)

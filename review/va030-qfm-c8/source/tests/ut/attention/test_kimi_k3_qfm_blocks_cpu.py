# SPDX-License-Identifier: Apache-2.0
"""CPU manager-block geometry contracts, not NPU/graph execution acceptance."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.ut.attention import test_kimi_k3_qfm_cpu as base

# Reuse the existing CPU scatter mock without importing its fixture by name.
qfm = base.qfm

MANAGER_BLOCKS = (128, 256, 384, 512, 1024, 2048)
KERNEL_BLOCK = 128
WIDTH = 576
TILE = 32
OFFSET = 256
MAIN_PAGES = 3
DRAFT_PAGES = 2


@pytest.mark.parametrize(
    "manager_blocks,alignment",
    (((128,), 32), ((128, 256), 64), ((384, 512), 384), ((1024, 2048), 512), ((384, 384), 96)),
)
@pytest.mark.parametrize("extra_bytes", (0, 1, 32))
def test_hybrid_common_page_alignment_is_minimal_and_valid_for_every_manager(
    qfm, manager_blocks, alignment, extra_bytes
):
    original = max(manager_blocks) * WIDTH + extra_bytes
    aligned = qfm.align_qfm_page_bytes(original, manager_blocks)
    assert aligned == ((original + alignment - 1) // alignment) * alignment
    assert aligned >= original and aligned - original < alignment
    assert qfm.align_qfm_page_bytes(aligned, manager_blocks) == aligned
    assert qfm.align_qfm_page_bytes(original, tuple(reversed(manager_blocks))) == aligned
    assert qfm.align_qfm_page_bytes(original, manager_blocks + manager_blocks) == aligned
    for manager_block in manager_blocks:
        raw = torch.zeros(2 * aligned, dtype=torch.uint8)
        cache = qfm.cache_view_for_manager(raw, manager_block, aligned)
        ratio = manager_block // KERNEL_BLOCK
        assert cache.shape[0] == 2 * ratio
        assert cache.stride(0) * ratio == aligned
        assert cache.stride(0) % TILE == 0


def test_common_page_alignment_without_qfm_managers_is_unchanged(qfm):
    assert qfm.align_qfm_page_bytes(123457, []) == 123457


@pytest.mark.parametrize("manager_block", MANAGER_BLOCKS)
@pytest.mark.parametrize("kernel_gap", (0, 32, 1024))
def test_manager_geometry_and_independent_scatter_guards(qfm, manager_block, kernel_gap):
    ratio = manager_block // KERNEL_BLOCK
    page_bytes = manager_block * WIDTH + ratio * kernel_gap
    main_bytes = MAIN_PAGES * page_bytes
    draft_start = OFFSET + main_bytes + OFFSET
    total = draft_start + DRAFT_PAGES * page_bytes + OFFSET
    backing = torch.full((total,), 0x5A, dtype=torch.uint8)
    backing[draft_start : draft_start + DRAFT_PAGES * page_bytes].fill_(0x31)
    raw = backing[OFFSET : OFFSET + main_bytes]
    cache = qfm.cache_view_for_manager(raw, manager_block, page_bytes)
    assert cache.dtype == torch.float8_e4m3fn
    assert cache.shape == (MAIN_PAGES * ratio, 1, WIDTH // TILE, KERNEL_BLOCK, TILE)
    assert cache.stride() == (page_bytes // ratio, KERNEL_BLOCK * WIDTH, KERNEL_BLOCK * TILE, TILE, 1)
    assert cache.storage_offset() == OFFSET
    assert cache.data_ptr() == raw.data_ptr()
    assert cache.untyped_storage().data_ptr() == backing.untyped_storage().data_ptr()

    # Distinct live slots cover kernel and manager boundaries without undefined
    # duplicate writes when manager_block itself is 128.
    live = list(
        dict.fromkeys(
            (
                0,
                127,
                128,
                manager_block - 1,
                manager_block,
                manager_block + 127,
                2 * manager_block - 1,
                2 * manager_block,
                MAIN_PAGES * manager_block - 1,
            )
        )
    )
    slots = torch.tensor([*live, -1, -17], dtype=torch.int64)
    rows = ((torch.arange(slots.numel() * WIDTH).reshape(-1, WIDTH) % 127) - 63).float() / 8
    scale = torch.tensor([1.0])
    encoded = rows.to(torch.float8_e4m3fn).view(torch.uint8)
    expected = backing.clone()
    for token, slot in enumerate(slots.tolist()):
        if slot < 0:
            continue
        manager, within_manager = divmod(slot, manager_block)
        subpage, within_kernel = divmod(within_manager, KERNEL_BLOCK)
        # Oracle starts from manager ownership, then its kernel subpage, then
        # the PA_NZ tile. It does not use the writer's flat tile-index formula.
        manager_start = OFFSET + manager * page_bytes
        subpage_start = manager_start + subpage * (page_bytes // ratio)
        for tile in range(WIDTH // TILE):
            start = subpage_start + tile * KERNEL_BLOCK * TILE + within_kernel * TILE
            expected[start : start + TILE] = encoded[token, tile * TILE : (tile + 1) * TILE]
    qfm.write_cache(rows, slots, cache, scale)
    assert torch.equal(backing, expected)
    assert (backing[:OFFSET] == 0x5A).all()
    assert (backing[draft_start : draft_start + DRAFT_PAGES * page_bytes] == 0x31).all()
    before = backing.clone()
    qfm.write_cache(rows, torch.full_like(slots, -1), cache, scale)
    assert torch.equal(backing, before)


@pytest.mark.parametrize(
    "manager_block,page_bytes,kernel_block,raw_adjust",
    (
        (64, 64 * WIDTH, 128, 0),
        (192, 192 * WIDTH, 128, 0),
        (0, 128 * WIDTH, 128, 0),
        (-128, 128 * WIDTH, 128, 0),
        (True, 128 * WIDTH, 128, 0),
        (512, 512 * WIDTH, 64, 0),
        (512, 512 * WIDTH + 32, 128, 0),
        (512, 512 * WIDTH - 128, 128, 0),
        (512, 512 * WIDTH, 128, -32),
    ),
)
def test_invalid_manager_geometry_is_rejected_without_writes(qfm, manager_block, page_bytes, kernel_block, raw_adjust):
    backing = torch.full((OFFSET + page_bytes + raw_adjust + OFFSET,), 0x45, dtype=torch.uint8)
    before = backing.clone()
    raw = backing[OFFSET:-OFFSET]
    with pytest.raises(ValueError):
        qfm.cache_view_for_manager(raw, manager_block, page_bytes, kernel_block_size=kernel_block)
    assert torch.equal(backing, before)


def test_manager_view_rejects_empty_and_unaligned_offset(qfm):
    page_bytes = 512 * WIDTH
    with pytest.raises(ValueError):
        qfm.cache_view_for_manager(torch.empty(0, dtype=torch.uint8), 512, page_bytes)
    backing = torch.zeros(page_bytes + 1, dtype=torch.uint8)
    with pytest.raises(ValueError):
        qfm.cache_view_for_manager(backing[1:], 512, page_bytes)


def selection_inputs(manager_block):
    vllm_config = SimpleNamespace(
        use_v2_model_runner=True,
        cache_config=SimpleNamespace(block_size=manager_block, cache_dtype="auto"),
        kv_transfer_config=None,
        speculative_config=SimpleNamespace(method="dspark"),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        # DSpark construction retains this target-derived config, so the
        # runner type alone does not identify its draft attention layers.
        model_config=SimpleNamespace(runner_type="generate"),
    )
    config = SimpleNamespace(
        enable_kimi_k3_qfm_c8=True,
        enable_sparse_sfa_c8=False,
        kvpp_config=SimpleNamespace(size=1),
        sparse_kv_offload_config=SimpleNamespace(enabled=False),
    )
    impl = SimpleNamespace(
        is_draft_model=vllm_config.model_config.runner_type == "draft",
        use_mla_rope=False,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        support_fp8_attention=True,
        num_kv_heads=1,
        num_heads=12,
        fa_quant_layer=False,
        enable_mlapo=False,
        enable_kv_nz=False,
        pcp_enabled=False,
        is_pcp_decode_sharded=False,
        vllm_config=vllm_config,
    )
    return impl, config


def set_model_tag(qfm, tag):
    qfm.importlib.import_module("vllm.compilation.backends").model_tag = tag


def load_dspark_cache_normalizer():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/kimi_k3_dspark.py"
    tree = ast.parse(path.read_text())
    node = next(
        item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "_normalize_dspark_mla_cache"
    )
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    scope = {"torch": torch}
    exec(compile(module, path, "exec"), scope)
    return scope[node.name]


def test_dspark_tag_excludes_target_derived_pre_rope_construction(qfm):
    impl, config = selection_inputs(512)
    assert impl.vllm_config.model_config.runner_type == "generate"
    assert not impl.is_draft_model and not impl.use_mla_rope
    set_model_tag(qfm, "dspark_head")
    assert not qfm.select_qfm(impl, config)


def test_select_qfm_reads_live_model_tag_and_restores_target_selection(qfm):
    impl, config = selection_inputs(512)
    set_model_tag(qfm, "dspark_head")
    assert not qfm.select_qfm(impl, config)
    set_model_tag(qfm, "backbone")
    assert qfm.select_qfm(impl, config)
    set_model_tag(qfm, "dspark_head")
    assert not qfm.select_qfm(impl, config)


def test_dspark_tag_precedes_target_only_geometry_validation(qfm):
    set_model_tag(qfm, "dspark_head")
    invalid_inputs = (
        lambda impl: setattr(impl, "num_heads", 3),
        lambda impl: setattr(impl.vllm_config.cache_config, "block_size", 192),
        lambda impl: setattr(impl, "enable_mlapo", True),
    )
    for invalidate in invalid_inputs:
        impl, config = selection_inputs(512)
        invalidate(impl)
        assert not qfm.select_qfm(impl, config)


@pytest.mark.parametrize("backend,expected", [("fia", True), ("qfm", False)])
def test_backend_selection_accepts_prolog(qfm, backend, expected):
    set_model_tag(qfm, "backbone")
    impl, config = selection_inputs(512)
    impl.enable_mlapo = True
    config.kimi_k3_c8_prefill_backend = backend
    assert qfm.select_qfm(impl, config)
    assert impl.qfm_fia_prefill is expected


def test_backend_rejects_unknown_value(qfm):
    set_model_tag(qfm, "backbone")
    impl, config = selection_inputs(512)
    config.kimi_k3_c8_prefill_backend = "automatic"
    with pytest.raises(ValueError, match="must be fia, qfa, qfm or fia_paged"):
        qfm.select_qfm(impl, config)


def test_backbone_still_rejects_invalid_target_geometry(qfm):
    set_model_tag(qfm, "backbone")
    invalid_inputs = (
        ("unsupported local query/KV heads", lambda impl: setattr(impl, "num_heads", 3)),
        (
            "positive multiple of 128",
            lambda impl: setattr(impl.vllm_config.cache_config, "block_size", 192),
        ),
        ("enable_kv_nz=false", lambda impl: setattr(impl, "enable_kv_nz", True)),
    )
    for message, invalidate in invalid_inputs:
        impl, config = selection_inputs(512)
        invalidate(impl)
        with pytest.raises(ValueError, match=message):
            qfm.select_qfm(impl, config)


def test_dspark_cache_normalizer_splits_real_4d_cache_and_preserves_tuple():
    normalize = load_dspark_cache_normalizer()
    cache = torch.arange(2 * 128 * 576, dtype=torch.float32).to(torch.bfloat16).view(2, 128, 1, 576)
    nope, rope = normalize(cache, 512)
    assert nope.shape == (2, 128, 1, 512)
    assert rope.shape == (2, 128, 1, 64) and rope.numel() > 0
    assert nope.dtype == rope.dtype == torch.bfloat16
    assert nope.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    assert rope.storage_offset() == cache.storage_offset() + 512
    torch.testing.assert_close(nope, cache[..., :512])
    torch.testing.assert_close(rope, cache[..., 512:])

    split_cache = (torch.empty(2, 128, 1, 512), torch.empty(2, 128, 1, 64))
    assert normalize(split_cache, 512) is split_cache


@pytest.mark.parametrize("manager_block", MANAGER_BLOCKS)
def test_select_qfm_admits_manager_multiples_but_excludes_draft(qfm, manager_block):
    impl, config = selection_inputs(manager_block)
    assert qfm.select_qfm(impl, config)
    impl.is_draft_model = True
    assert not qfm.select_qfm(impl, config)


@pytest.mark.parametrize("manager_block", (64, 192))
def test_select_qfm_rejects_non_kernel_multiples(qfm, manager_block):
    impl, config = selection_inputs(manager_block)
    with pytest.raises(ValueError):
        qfm.select_qfm(impl, config)


def prepared_metadata(qfm, monkeypatch, manager=256, gap=256, mlapo=True, prefill_mlapo=False, fia=False):
    impl, _ = selection_inputs(manager)
    impl.enable_mlapo = mlapo
    impl.qfm_mlapo_prefill = prefill_mlapo
    impl.qfm_fia_prefill = fia
    impl.vllm_config.model_config.max_model_len = 2048
    impl.vllm_config.parallel_config.cp_kv_cache_interleave_size = 16
    ratio = manager // KERNEL_BLOCK
    page_bytes = ratio * (KERNEL_BLOCK * WIDTH + gap)
    spec = SimpleNamespace(block_size=manager, page_size_bytes=page_bytes, page_size_padded=page_bytes)
    monkeypatch.setattr(qfm, "load_ops", lambda: (None, lambda lengths, *args, **kwargs: lengths.clone()))
    return qfm.QFMMetadataBuilder(impl, torch.device("cpu"), spec)


@pytest.mark.parametrize("manager,gap", [(128, 0), (256, 256), (768, 1024)])
def test_shared_scatter_plan_reused_across_layers_without_new_indices(qfm, monkeypatch, manager, gap):
    b = prepared_metadata(qfm, monkeypatch, manager, gap, mlapo=False)
    c = base.common([4, 5], [132, 133], capacity=12)
    c.slot_mapping = torch.tensor([128, 3, 255, 0, 130, 127, 1, 129, 2])
    batch = b.build(c, retain_for_graph=True)
    stride = 73728 + gap
    assert batch.scatter_plan.stride == stride
    assert torch.equal(batch.scatter_plan.offsets, qfm.build_scatter_offsets(batch.slots, stride))
    raw = torch.full((256 + 2 * stride + 256,), 0x6B, dtype=torch.uint8)
    cache = qfm.cache_view(raw[256 : 256 + 2 * stride], 2, stride)
    reference = raw.clone()
    rows = torch.randn(12, WIDTH)
    qfm.write_cache(rows, batch.slots, qfm.cache_view(reference[256 : 256 + 2 * stride], 2, stride), torch.ones(1))
    offsets_ptr = batch.scatter_plan.offsets.data_ptr()

    def forbidden(*args, **kwargs):
        raise AssertionError("Layer writer must reuse group offsets")

    monkeypatch.setattr(qfm, "build_scatter_offsets", forbidden)
    monkeypatch.setattr(torch, "arange", forbidden)
    for _ in range(3):
        qfm.write_cache(rows, batch.slots, cache, torch.ones(1), scatter_plan=batch.scatter_plan)
        assert torch.equal(raw, reference)
        assert batch.scatter_plan.offsets.data_ptr() == offsets_ptr
    qfm.write_cache(rows[4:], batch.slots[4:], cache, torch.ones(1), scatter_plan=batch.scatter_plan.slice(4, 12))
    assert torch.equal(raw, reference)


def test_retained_shared_buffers_refresh_without_rebinding(qfm, monkeypatch):
    b = prepared_metadata(qfm, monkeypatch, fia=True)
    c = base.common([4], [132], capacity=8)
    batch = b.build(c, retain_for_graph=True)
    ptrs = (
        batch.scatter_plan.offsets.data_ptr(),
        batch.invalid.data_ptr(),
        batch.invisible.data_ptr(),
        batch.prolog_kwargs["staging_indices"].data_ptr(),
        batch.fia_mask.data_ptr(),
    )
    c.seq_lens[:] = 0
    c.slot_mapping = torch.tensor([255, 128])
    updated = b.build(c)
    assert updated is batch
    assert ptrs == (
        batch.scatter_plan.offsets.data_ptr(),
        batch.invalid.data_ptr(),
        batch.invisible.data_ptr(),
        batch.prolog_kwargs["staging_indices"].data_ptr(),
        batch.fia_mask.data_ptr(),
    )
    assert batch.invalid.all() and batch.invisible.all()
    assert (batch.scatter_plan.offsets == -1).all()
    c.seq_lens[:] = 132
    b.build(c)
    assert torch.equal(batch.invalid, ~batch.live)
    assert torch.equal(batch.invisible, ~batch.visible)
    assert torch.equal(batch.scatter_plan.offsets, qfm.build_scatter_offsets(batch.slots, b.cache_stride))
    assert torch.equal(batch.prolog_kwargs["staging_indices"], torch.arange(8))
    assert batch.fia_mask.dtype == torch.bool
    assert torch.equal(batch.fia_mask, b.mask.bool())


@pytest.mark.parametrize("wrong", ["stride", "slots", "slots_shape", "dtype", "shape", "contiguous", "device"])
def test_shared_scatter_plan_fallback_uses_live_geometry_on_each_replay(qfm, wrong):
    stride = 73728 + 256
    slots = torch.tensor([0, -1, 129])
    backing = torch.full((256 + 2 * stride + 256,), 0x5A, dtype=torch.uint8)
    cache = qfm.cache_view(backing[256 : 256 + 2 * stride], 2, stride)
    offsets = qfm.build_scatter_offsets(slots, stride)
    plan = qfm.QFMScatterPlan(slots, offsets, stride)
    if wrong == "stride":
        plan.stride += 32
    elif wrong == "slots":
        plan.slots = torch.tensor([255, 128, 127])
    elif wrong == "slots_shape":
        plan.slots = slots.view(1, -1)
    elif wrong == "dtype":
        plan.offsets = offsets.int()
    elif wrong == "shape":
        plan.offsets = offsets[:, :1]
    elif wrong == "contiguous":
        plan.offsets = offsets.t().contiguous().t()
    elif wrong == "device":
        plan.offsets = torch.empty(offsets.shape, dtype=torch.int64, device="meta")
    values = torch.randn(3, WIDTH).to(torch.float8_e4m3fn)
    reference = backing.clone()
    expected_cache = qfm.cache_view(reference[256 : 256 + 2 * stride], 2, stride)
    qfm.scatter_fp8_cache(values, slots, expected_cache)
    with pytest.warns(RuntimeWarning, match="Scatter plan fallback"):
        qfm.scatter_fp8_cache(values, slots, cache, scatter_plan=plan)
    assert torch.equal(backing, reference)
    # Stand in for a new graph replay: refresh slots in place. The fallback
    # must recompute from them, rather than retain the first call's offsets.
    slots.copy_(torch.tensor([255, 128, -1]))
    qfm.scatter_fp8_cache(values, slots, expected_cache)
    qfm.scatter_fp8_cache(values, slots, cache, scatter_plan=plan)
    assert torch.equal(backing, reference)


def test_direct_mode2_decode_does_not_prepare_unused_scatter(qfm, monkeypatch):
    b = prepared_metadata(qfm, monkeypatch, gap=0, mlapo=True, prefill_mlapo=False)
    c = base.common([4], [132], capacity=8)
    c.is_prefilling.fill_(False)

    def forbidden(*args, **kwargs):
        raise AssertionError("Direct mode2 Decode does not use Scatter offsets")

    monkeypatch.setattr(qfm, "build_scatter_offsets", forbidden)
    batch = b.build(c, retain_for_graph=True)
    assert not batch.has_prefill
    assert "staging_indices" not in batch.prolog_kwargs


def expanded_table(manager_ids, ratio=4):
    """Stand in for MRV2's already-completed manager-to-kernel conversion."""
    return torch.tensor(
        [[manager * ratio + subpage for manager in row for subpage in range(ratio)] for row in manager_ids],
        dtype=torch.int32,
    )


def metadata_common(queries, lengths, capacity, manager_ids):
    common = base.common(queries, lengths, capacity)
    common.block_table_tensor = expanded_table(manager_ids)
    common.slot_mapping = torch.arange(sum(queries), dtype=torch.int64) + 2 * 512
    if common.slot_mapping.numel() > 2:
        common.slot_mapping[2] = -1
    return common


def test_apc_dspark_padding_uses_already_expanded_kernel_ids_once(qfm, monkeypatch):
    builder, calls = base.builder(qfm, monkeypatch)
    common = metadata_common([1, 17, 9], [513, 1025, 767], 48, [[2, 4], [3, 5], [6, 7]])
    original = common.block_table_tensor.clone()
    result = builder.build(common)
    assert result.used.tolist() == [1, 16, 1, 9, 0, 0]
    assert result.lengths.tolist() == [513, 1024, 1025, 767, 0, 0]
    assert result.table.shape == (6, 8)
    assert result.table[0, :4].tolist() == [8, 9, 10, 11]
    assert torch.equal(result.table, original[[0, 1, 1, 2, 2, 2]])
    assert torch.equal(common.block_table_tensor, original)
    assert torch.equal(result.slots[:27], common.slot_mapping)
    assert (result.slots[27:] == -1).all()
    assert result.mask_mode == 3
    assert len(calls) == 1


@pytest.mark.parametrize("rank", (0, 1))
def test_dcp_per_token_prefix_preserves_expanded_kernel_ids(qfm, monkeypatch, rank):
    builder, _ = base.builder(qfm, monkeypatch, dcp=2, rank=rank)
    common = metadata_common([9, 4], [513, 1025], 16, [[2, 4], [3, 5]])
    result = builder.build(common)
    expected = [
        sum((position // 16) % 2 == rank for position in range(length))
        for length in [*range(505, 514), *range(1022, 1026)]
    ]
    assert result.lengths.tolist() == expected + [0, 0, 0]
    assert result.table.shape == (16, 8)
    assert torch.equal(result.table, common.block_table_tensor[[0] * 9 + [1] * 7])
    assert result.table[0, :4].tolist() == [8, 9, 10, 11]
    assert torch.equal(result.slots[:13], common.slot_mapping)
    assert (result.slots[13:] == -1).all()
    assert result.mask_mode == 0 and result.max_q == 1


@pytest.mark.parametrize("dcp,rank", ((1, 0), (2, 1)))
def test_graph_metadata_refresh_keeps_addresses_and_does_not_expand_twice(qfm, monkeypatch, dcp, rank):
    builder, calls = base.builder(qfm, monkeypatch, dcp=dcp, rank=rank)
    first = builder.build(metadata_common([17, 4], [513, 1025], 32, [[2, 6], [3, 7]]), retain_for_graph=True)
    fields = ("cu", "used", "lengths", "slots", "table", "live", "visible", "schedule")
    addresses = {field: getattr(first, field).data_ptr() for field in fields}
    refreshes = (([4, 9], [516, 1020]), ([0, 0], [0, 0]), ([1, 8], [1025, 129]))
    for queries, lengths in refreshes:
        common = metadata_common(queries, lengths, 32, [[5, 8], [9, 11]])
        common.seq_lens_cpu = torch.tensor([9999, 9999])  # Deliberately stale DSpark-side mirror.
        result = builder.build(common)
        assert result is first
        assert addresses == {field: getattr(result, field).data_ptr() for field in fields}
        assert result.cu.tolist() == list(range(33))
        actual = sum(queries)
        assert result.used.tolist() == [1] * actual + [0] * (32 - actual)
        owners = [0] * queries[0] + [1] * (32 - queries[0])
        assert torch.equal(result.table, common.block_table_tensor[owners])
        assert torch.equal(result.slots[:actual], common.slot_mapping)
        assert (result.slots[actual:] == -1).all()
        expected = []
        for count, length in zip(queries, lengths):
            for prefix in range(length - count + 1, length + 1):
                expected.append(prefix if dcp == 1 else sum((p // 16) % dcp == rank for p in range(prefix)))
        assert result.lengths.tolist() == expected + [0] * (32 - actual)
        assert torch.equal(result.schedule, result.lengths)
        assert result.table.shape[1] == 8
    assert len(calls) == len(refreshes) + 1

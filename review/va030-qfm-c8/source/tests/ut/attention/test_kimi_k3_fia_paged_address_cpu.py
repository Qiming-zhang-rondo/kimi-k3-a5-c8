# SPDX-License-Identifier: Apache-2.0
"""Independent sidecar byte-address/COW contracts; no native NPU acceptance."""

import ast
import importlib.util
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.ut.attention import test_kimi_k3_qfm_cpu as base

qfm = base.qfm
BLOCK = 128
FP8_PAGE = BLOCK * 576
NOPE_PAGE = BLOCK * 512 * 2
RAW_PAGE = BLOCK * 64 * 2
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def paged(qfm, monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm_ascend.attention.kimi_k3_qfm", qfm)
    path = ROOT / "vllm_ascend/attention/kimi_k3_fia_paged.py"
    spec = importlib.util.spec_from_file_location("paged_fia_address_cpu_under_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "vllm_ascend.attention.kimi_k3_fia_paged", module)
    return module


def allocation(qfm, pages, offset=256, guard=0x5A, gap=0):
    stride = FP8_PAGE + gap
    backing = torch.full((offset + pages * stride + 512,), guard, dtype=torch.uint8)
    cache = qfm.cache_view(backing[offset : offset + pages * stride], pages, stride)
    parts = []
    for width in (512, 64):
        page_bytes = BLOCK * width * 2
        side = torch.full((offset + pages * page_bytes + 512,), guard, dtype=torch.uint8)
        tensor = side[offset : offset + pages * page_bytes].view(torch.bfloat16).view(pages, 1, width // 16, BLOCK, 16)
        parts.append(SimpleNamespace(backing=side, offset=offset, page_bytes=page_bytes, tensor=tensor))
    cache._qfm_bf16_prefill = True
    cache._qfm_bf16_views = tuple(part.tensor for part in parts)
    return SimpleNamespace(backing=backing, cache=cache, pages=pages, offset=offset, stride=stride, parts=parts)


def snapshot(storage):
    return tuple(part.clone() for part in (storage.backing, *(part.backing for part in storage.parts)))


def assert_bytes(storage, expected):
    for actual, golden in zip((storage.backing, *(part.backing for part in storage.parts)), expected):
        assert torch.equal(actual, golden)


def bf16_addresses(slots, width, base_offset):
    """Resolve bytes independently of all production view/scatter helpers."""
    slots = torch.as_tensor(slots, dtype=torch.int64)
    features = torch.arange(width)
    words = (
        base_offset
        + slots[:, None].div(BLOCK, rounding_mode="floor") * BLOCK * width * 2
        + features[None, :].div(16, rounding_mode="floor") * BLOCK * 16 * 2
        + slots[:, None].remainder(BLOCK) * 16 * 2
        + features[None, :].remainder(16) * 2
    )
    return words[:, :, None] + torch.arange(2)


def overwrite_oracle(storage, expected, slots, rows):
    keep = slots >= 0
    slots, rows = slots[keep], rows[keep]
    for index, (width, values) in enumerate(((512, rows[:, :512]), (64, rows[:, 512:])), 1):
        addresses = bf16_addresses(slots, width, storage.parts[index - 1].offset)
        expected[index][addresses] = values.contiguous().view(torch.uint8).reshape(len(slots), width, 2)


@pytest.mark.parametrize("pages,offset,gap", [(1, 0, 0), (3, 256, 256), (6, 512, 1024)])
def test_dense_sidecars_are_separate_from_shared_main_storage(paged, qfm, pages, offset, gap):
    storage = allocation(qfm, pages, offset, gap=gap)
    nope, raw = paged.sidecar_views(storage.cache)
    assert nope.shape == (pages, 1, 32, 128, 16)
    assert raw.shape == (pages, 1, 4, 128, 16)
    assert nope.dtype == raw.dtype == torch.bfloat16
    assert nope.is_contiguous() and raw.is_contiguous()
    assert nope.untyped_storage().data_ptr() != storage.cache.untyped_storage().data_ptr()
    assert raw.untyped_storage().data_ptr() != storage.cache.untyped_storage().data_ptr()
    assert nope.data_ptr() - storage.parts[0].backing.data_ptr() == offset
    assert raw.data_ptr() - storage.parts[1].backing.data_ptr() == offset


@pytest.mark.parametrize("pages,offset,gap", [(1, 256, 0), (3, 512, 256), (6, 256, 1024)])
def test_current_bf16_writer_preserves_fp8_and_nonlive_bytes(paged, qfm, pages, offset, gap):
    storage = allocation(qfm, pages, offset, gap=gap)
    slots = torch.tensor([pages * BLOCK - 1, 0, -1, 127, min(128, pages * BLOCK - 2)])
    rows = ((torch.arange(len(slots) * 576).reshape(-1, 576) % 251) / 8).to(torch.bfloat16)
    expected = snapshot(storage)
    overwrite_oracle(storage, expected, slots, rows)
    paged.write_bf16(storage.cache, rows, slots)
    assert_bytes(storage, expected)
    before = snapshot(storage)
    paged.write_bf16(storage.cache, rows, torch.full_like(slots, -1))
    assert_bytes(storage, before)


@pytest.mark.parametrize("pages,offset,gap", [(1, 256, 0), (3, 512, 256), (6, 256, 1024)])
def test_fp8_mirror_uses_live_slots_and_actual_encoded_values(paged, qfm, pages, offset, gap):
    storage = allocation(qfm, pages, offset, gap=gap)
    slots = torch.tensor([pages * BLOCK - 1, -1, 0, 127])
    codes = ((torch.arange(len(slots) * 576).reshape(-1, 576) * 7 + 3) % 112).to(torch.uint8)
    features = torch.arange(576)
    for index, slot in enumerate(slots.tolist()):
        if slot >= 0:
            page, token = divmod(slot, BLOCK)
            addresses = (
                storage.offset + page * storage.stride + features // 32 * BLOCK * 32 + token * 32 + features % 32
            )
            storage.backing[addresses] = codes[index]
    expected = snapshot(storage)
    overwrite_oracle(storage, expected, slots, codes.view(torch.float8_e4m3fn).to(torch.bfloat16))
    paged.mirror_fp8_slots(storage.cache, slots)
    assert_bytes(storage, expected)


@pytest.mark.parametrize("width", [512, 64])
def test_noncontiguous_bf16_pool_is_rejected_before_native_fia(paged, qfm, width):
    storage = allocation(qfm, 3)
    source = torch.zeros(3 * BLOCK * width * 2, dtype=torch.bfloat16)
    shape = (3, 1, width // 16, BLOCK, 16)
    strides = (BLOCK * width + 16, BLOCK * width, BLOCK * 16, 16, 1)
    gapped = torch.as_strided(source, shape, strides)
    parts = list(storage.cache._qfm_bf16_views)
    parts[0 if width == 512 else 1] = gapped
    storage.cache._qfm_bf16_views = tuple(parts)
    with pytest.raises(ValueError, match="dense"):
        paged.sidecar_views(storage.cache)


def test_missing_sidecar_allocation_is_rejected_before_any_write(paged, qfm):
    storage = allocation(qfm, 3)
    del storage.cache._qfm_bf16_views
    before = snapshot(storage)
    with pytest.raises(ValueError, match="budgeted"):
        paged.sidecar_views(storage.cache)
    assert_bytes(storage, before)


def test_sidecar_cannot_hide_inside_another_hybrid_groups_shared_storage(paged, qfm):
    pages, offset = 3, 256
    backing = torch.zeros(offset + pages * (FP8_PAGE + NOPE_PAGE + RAW_PAGE), dtype=torch.uint8)
    cache = qfm.cache_view(backing[offset : offset + pages * FP8_PAGE], pages, FP8_PAGE)
    # These ranges do not overlap the exposed FP8 tensor, but still belong to
    # the common allocation where other hybrid groups own live state pages.
    start = offset + pages * FP8_PAGE
    nope = backing[start : start + pages * NOPE_PAGE].view(torch.bfloat16).view(pages, 1, 32, 128, 16)
    start += pages * NOPE_PAGE
    raw = backing[start : start + pages * RAW_PAGE].view(torch.bfloat16).view(pages, 1, 4, 128, 16)
    cache._qfm_bf16_prefill = True
    cache._qfm_bf16_views = nope, raw
    with pytest.raises(ValueError, match="independent"):
        paged.sidecar_views(cache)


def extract_definition(relative_path, name, globals_dict):
    tree = ast.parse((ROOT / relative_path).read_text())
    node = next(node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(relative_path), "exec"), globals_dict)
    return globals_dict[name]


@pytest.mark.parametrize("ratio,offset,gap", [(1, 256, 0), (2, 512, 256), (6, 256, 1024)])
@pytest.mark.parametrize("logical_blocks", [2, 3])
def test_apc_partial_prefix_cow_copies_both_sidecar_pools(paged, qfm, ratio, offset, gap, logical_blocks):
    storage = allocation(qfm, logical_blocks * ratio, offset, gap=gap)
    allocations = [(storage.backing, storage.stride), *((part.backing, part.page_bytes) for part in storage.parts)]
    for backing, stride in allocations:
        payload = backing[offset : offset + storage.pages * stride]
        payload.copy_((torch.arange(payload.numel()) * 17 % 251).to(torch.uint8))
    expected = snapshot(storage)
    # Snapshot sources before a 0->1, 1->2 chain. Main-page padding belongs
    # to other caches and must remain unchanged; BF16 pools are dense.
    for index, ((backing, stride), page_bytes) in enumerate(zip(allocations, (FP8_PAGE, NOPE_PAGE, RAW_PAGE))):
        for source, target in ((0, 1), *([(1, 2)] if logical_blocks > 2 else [])):
            for subpage in range(ratio):
                src = offset + (source * ratio + subpage) * stride
                dst = offset + (target * ratio + subpage) * stride
                expected[index][dst : dst + page_bytes] = backing[src : src + page_bytes]
    copy_blocks = extract_definition("vllm_ascend/worker/utils.py", "copy_kv_cache_blocks_inplace", {"torch": torch})
    copies = [SimpleNamespace(src_block_id=0, dst_block_id=1)]
    if logical_blocks > 2:
        copies.append(SimpleNamespace(src_block_id=1, dst_block_id=2))
    copy_blocks([storage.cache, storage.cache], logical_blocks, copies)
    assert_bytes(storage, expected)


def test_v2_zeroer_receives_all_three_dense_pools_only_for_opted_cache(paged, qfm):
    captures = []

    class Zeroer:
        def __init__(self, device, **kwargs):
            self.layers = kwargs["static_forward_context"]
            captures.append(self.layers)

        def zero_block_ids(self, block_ids):
            for layer in self.layers.values():
                for block_id in block_ids:
                    layer.kv_cache[block_id].zero_()

    @dataclass
    class Group:
        layer_names: list

    storage = allocation(qfm, 3)
    ordinary = torch.full((3, 128, 128), 7.0)
    cls = extract_definition(
        "vllm_ascend/worker/v2/utils.py",
        "AscendV2KVBlockZeroer",
        {"torch": torch, "KVBlockZeroer": Zeroer, "replace": replace, "SimpleNamespace": SimpleNamespace},
    )
    zeroer = cls(
        torch.device("cpu"),
        attn_groups_iter=[Group(["paged", "ordinary"])],
        kernel_block_sizes=[128],
        static_forward_context={
            "paged": SimpleNamespace(kv_cache=storage.cache),
            "ordinary": SimpleNamespace(kv_cache=ordinary),
        },
        num_blocks=3,
    )
    assert len(captures) == 3
    assert set(captures[0]) == {"paged", "ordinary"}
    assert [set(value) for value in captures[1:]] == [{"paged"}, {"paged"}]
    actual = [value["paged"].kv_cache for value in captures]
    assert [value.data_ptr() for value in actual] == [
        storage.cache.data_ptr(),
        *(part.tensor.data_ptr() for part in storage.parts),
    ]
    expected = snapshot(storage)
    for index, (_, stride) in enumerate(
        [(storage.backing, storage.stride), *((part.backing, part.page_bytes) for part in storage.parts)]
    ):
        page_bytes = (FP8_PAGE, NOPE_PAGE, RAW_PAGE)[index]
        expected[index][storage.offset + stride : storage.offset + stride + page_bytes] = 0
    zeroer.zero_block_ids([1])
    assert_bytes(storage, expected)
    assert ordinary[0].eq(7).all() and ordinary[1].eq(0).all() and ordinary[2].eq(7).all()


def test_unflagged_fp8_cow_never_discovers_sidecars_from_storage_capacity(paged, qfm):
    storage = allocation(qfm, 3)
    del storage.cache._qfm_bf16_prefill
    payload = storage.backing[storage.offset : storage.offset + 3 * FP8_PAGE]
    payload.copy_((torch.arange(payload.numel()) * 19 % 251).to(torch.uint8))
    expected = snapshot(storage)
    expected[0][storage.offset + FP8_PAGE : storage.offset + 2 * FP8_PAGE] = expected[0][
        storage.offset : storage.offset + FP8_PAGE
    ].clone()
    copy_blocks = extract_definition("vllm_ascend/worker/utils.py", "copy_kv_cache_blocks_inplace", {"torch": torch})
    copy_blocks([storage.cache], 3, [SimpleNamespace(src_block_id=0, dst_block_id=1)])
    assert_bytes(storage, expected)

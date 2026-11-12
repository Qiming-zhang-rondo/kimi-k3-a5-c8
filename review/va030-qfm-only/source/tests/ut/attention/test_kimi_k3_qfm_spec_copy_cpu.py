# SPDX-License-Identifier: Apache-2.0
"""Exercise the real spec-copy chain without importing the NPU framework.

The production Ascend class and upstream dataclasses/methods are AST-loaded
unchanged; only unrelated package imports/registries are omitted. CPU Scatter
stands in for the native writer, not for address or spec-copy arithmetic.
"""

import ast
import importlib.util
import math
import sys
from dataclasses import dataclass, fields, replace
from enum import Enum, IntEnum
from fractions import Fraction
from functools import cached_property
from pathlib import Path
from types import ModuleType
from typing import Self

import pytest
import torch

from tests.ut.attention import test_kimi_k3_qfm_cpu as base

ROOT = Path(__file__).resolve().parents[3]
ASCEND_SOURCE = ROOT / "vllm_ascend/core/kv_cache_interface.py"
qfm = base.qfm


def _upstream_source():
    try:
        package = importlib.util.find_spec("vllm")
    except (ModuleNotFoundError, ValueError):
        package = None
    if package is not None and package.submodule_search_locations:
        return Path(next(iter(package.submodule_search_locations))) / "v1/kv_cache_interface.py"
    # Local lightweight CPU environment: use the checked-out dependency.
    source = ROOT.parent / "vllm-v0.30.0/vllm/v1/kv_cache_interface.py"
    if not source.is_file():
        pytest.skip("An installed or checked-out vLLM source is required for actual spec-copy coverage")
    return source


@pytest.fixture
def spec_type(monkeypatch):
    module = ModuleType("qfm_spec_copy_production_types")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    module.__dict__.update(
        dataclass=dataclass,
        fields=fields,
        replace=replace,
        Enum=Enum,
        IntEnum=IntEnum,
        Fraction=Fraction,
        cached_property=cached_property,
        torch=torch,
        Self=Self,
        cdiv=lambda a, b: (a + b - 1) // b,
        round_up=lambda a, b: math.ceil(a / b) * b,
        get_dtype_size=lambda dtype: torch.empty(0, dtype=dtype).element_size(),
    )
    upstream = _upstream_source()
    names = {
        "KVQuantMode",
        "KVCacheSpecKind",
        "SparseCacheRole",
        "KVCacheSpec",
        "AttentionSpec",
        "FullAttentionSpec",
        "MLAAttentionSpec",
        "UniformTypeKVCacheSpecs",
        "_apply_alignment_padding",
    }
    nodes = [node for node in ast.parse(upstream.read_text()).body if getattr(node, "name", None) in names]
    ascend_names = {"get_kv_cache_compression_ratio", "get_storage_block_size", "AscendMLAAttentionSpec"}
    ascend = [node for node in ast.parse(ASCEND_SOURCE.read_text()).body if getattr(node, "name", None) in ascend_names]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, *nodes, *ascend], type_ignores=[]))
    exec(compile(tree, str(ASCEND_SOURCE), "exec"), module.__dict__)
    return module.AscendMLAAttentionSpec


def _spec(spec_type, block_size, padded=None, qfm_enabled=True):
    return spec_type(
        block_size=block_size,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.float8_e4m3fn,
        cache_qfm_c8=qfm_enabled,
        page_size_padded=padded,
    )


def test_reported_768_manager_page_becomes_147456_kernel_stride(spec_type, qfm):
    original = _spec(spec_type, 768, 884736)
    copied = original.copy_with_new_block_size(128)
    cache = qfm.cache_view_for_manager(torch.empty(2 * original.page_size_bytes, dtype=torch.uint8), 768, 884736)
    assert copied.block_size == 128
    assert copied.page_size_padded == copied.page_size_bytes == cache.stride(0) == 147456
    assert original.block_size == 768 and original.page_size_padded == 884736


@pytest.mark.parametrize("ratio", [1, 2, 4, 6])
@pytest.mark.parametrize("kernel_padding", [0, 256])
@pytest.mark.parametrize("offset", [0, 256])
@pytest.mark.parametrize("writer", ["ordinary", "staged_fp8"])
def test_copied_spec_shares_live_scatter_plan_across_layers_and_refresh(
    spec_type, qfm, monkeypatch, ratio, kernel_padding, offset, writer
):
    stride = 128 * 576 + kernel_padding
    manager = 128 * ratio
    original = _spec(spec_type, manager, stride * ratio)
    copied = original.copy_with_new_block_size(128)
    backing = torch.full((offset + 2 * original.page_size_bytes + 256,), 0x5A, dtype=torch.uint8)
    cache = qfm.cache_view_for_manager(
        backing[offset : offset + 2 * original.page_size_bytes], manager, original.page_size_bytes
    )
    impl = base.impl_config()
    impl.enable_mlapo = True
    monkeypatch.setattr(qfm, "load_ops", lambda: (None, lambda lengths, *a, **kw: lengths.clone()))
    builder = qfm.QFMMetadataBuilder(impl, torch.device("cpu"), copied)
    assert builder.cache_stride == cache.stride(0) == stride
    common = base.common([4], [260], capacity=8)
    common.slot_mapping = torch.tensor([128, -1, 0, 2 * ratio * 128 - 1])
    batch = builder.build(common, retain_for_graph=True)
    slots_pointer = batch.slots.data_ptr()
    if kernel_padding:
        assert batch.scatter_plan is not None
        offsets_pointer = batch.scatter_plan.offsets.data_ptr()
    else:
        # Dense pages are written directly by DSL Prolog in production; no
        # extra Scatter plan or scratch buffer belongs in the captured step.
        assert batch.scatter_plan is None
        offsets_pointer = None
    rows = torch.randn(8, 576)
    encoded = rows.clamp(-448, 448).to(torch.float8_e4m3fn)
    expected = backing.clone()

    def forbid_rebuild(*args, **kwargs):
        raise AssertionError("A layer writer must reuse the corrected shared plan without fallback")

    for refresh in (False, True):
        if refresh:
            common.slot_mapping = torch.tensor([127, 0, 129, -1])
            assert builder.build(common) is batch
        assert batch.slots.data_ptr() == slots_pointer
        if offsets_pointer is not None:
            assert batch.scatter_plan.offsets.data_ptr() == offsets_pointer
        else:
            assert batch.scatter_plan is None
        for row, slot in zip(encoded.view(torch.uint8), batch.slots.tolist()):
            if slot < 0:
                continue
            page, token = divmod(slot, 128)
            for tile in range(18):
                start = offset + page * stride + tile * 128 * 32 + token * 32
                expected[start : start + 32] = row[tile * 32 : (tile + 1) * 32]
        with monkeypatch.context() as guarded:
            if kernel_padding:
                guarded.setattr(qfm, "build_scatter_offsets", forbid_rebuild)
            guarded.setattr(qfm.warnings, "warn", forbid_rebuild)
            for _ in range(3):
                if writer == "ordinary":
                    # Exercise the standalone writer's address contract even
                    # though model forward always uses fused DSL Prolog.
                    qfm.write_cache(rows, batch.slots, cache, torch.ones(1), scatter_plan=batch.scatter_plan)
                else:
                    # Payload from a Prolog scratch writer is already FP8.
                    qfm.scatter_fp8_cache(encoded, batch.slots, cache, scatter_plan=batch.scatter_plan)
                assert torch.equal(backing, expected)
    assert original.block_size == manager and original.page_size_padded == stride * ratio


@pytest.mark.parametrize("ratio", [1, 2, 4, 6])
def test_unpadded_copy_uses_recomputed_payload_size(spec_type, ratio):
    original = _spec(spec_type, 128 * ratio)
    copied = original.copy_with_new_block_size(128)
    assert copied.page_size_padded is None
    assert copied.page_size_bytes == 128 * 576
    assert original.page_size_bytes == ratio * copied.page_size_bytes


@pytest.mark.parametrize("qfm_enabled", [False, True])
@pytest.mark.parametrize("new_block", [768, 1536])
def test_same_or_larger_block_keeps_upstream_padding_semantics(spec_type, qfm_enabled, new_block):
    original = _spec(spec_type, 768, 884736, qfm_enabled)
    copied = original.copy_with_new_block_size(new_block)
    assert copied.block_size == new_block and copied.page_size_padded == 884736
    assert copied == super(spec_type, original).copy_with_new_block_size(new_block)
    assert copied is not original
    assert original.block_size == 768 and original.page_size_padded == 884736


def test_non_qfm_shrink_keeps_upstream_padding_semantics(spec_type):
    original = _spec(spec_type, 768, 884736, qfm_enabled=False)
    copied = original.copy_with_new_block_size(128)
    assert copied.page_size_padded == 884736
    assert copied == super(spec_type, original).copy_with_new_block_size(128)


@pytest.mark.parametrize("padded,new_block", [(884737, 128), (884742, 128), (884736, 0), (884736, -128)])
def test_invalid_qfm_subpage_alignment_is_rejected(spec_type, padded, new_block):
    original = _spec(spec_type, 768, padded)
    with pytest.raises(ValueError, match="QFM"):
        original.copy_with_new_block_size(new_block)
    assert original.block_size == 768 and original.page_size_padded == padded

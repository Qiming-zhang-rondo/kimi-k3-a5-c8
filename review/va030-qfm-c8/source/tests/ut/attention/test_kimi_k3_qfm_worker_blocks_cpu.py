# SPDX-License-Identifier: Apache-2.0
"""Bounded CPU hookup checks for K3 QFM manager/kernel cache blocks."""

import ast
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.ut.attention.test_kimi_k3_qfm_cpu import qfm as existing_qfm_fixture

ROOT = Path(__file__).resolve().parents[3]
ATTN_UTILS = ROOT / "vllm_ascend/worker/v2/attn_utils.py"
QFM_MODULE = "vllm_ascend.attention.kimi_k3_qfm"


@dataclass(frozen=True)
class PageSpec:
    block_size: int
    payload_bytes: int
    cache_qfm_c8: bool = False
    page_size_padded: int | None = None

    @property
    def page_size_bytes(self):
        return self.page_size_padded if self.page_size_padded is not None else self.payload_bytes


class FullAttentionSpec:
    pass


class SlidingWindowSpec:
    pass


class AscendSFAIndexerCacheSpec:
    pass


@pytest.fixture
def qfm(monkeypatch):
    return existing_qfm_fixture.__wrapped__(monkeypatch)


def _source_function(name):
    tree = ast.parse(ATTN_UTILS.read_text(encoding="utf-8"))
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _compile_function(node, scope):
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    exec(compile(module, str(ATTN_UTILS), "exec"), scope)
    return scope[node.name]


def _shared_mamba_alignment():
    source = _source_function("get_kv_cache_spec")
    branch = next(
        node
        for node in source.body
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "mamba_specs"
    )
    wrapper = ast.FunctionDef(
        name="run_shared_mamba_alignment",
        args=ast.arguments(
            posonlyargs=[],
            args=[
                ast.arg(arg="kv_cache_spec"),
                ast.arg(arg="mamba_specs"),
                ast.arg(arg="attention_layer_names"),
            ],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[branch, ast.Return(value=ast.Name(id="kv_cache_spec", ctx=ast.Load()))],
        decorator_list=[],
    )
    scope = {
        "replace": replace,
        "FullAttentionSpec": FullAttentionSpec,
        "SlidingWindowSpec": SlidingWindowSpec,
    }
    _compile_function(_source_function("_align_hybrid_attention_page_sizes"), scope)
    return _compile_function(wrapper, scope)


def _reshape_qfm_branch():
    logs = []
    scope = {
        "torch": torch,
        "get_current_vllm_config": lambda: SimpleNamespace(),
        "_is_dsv4_model": lambda _config: False,
        "_get_layer_kv_cache_specs": lambda config: config.layer_specs,
        "is_deepseek_v41_cache": lambda _specs: False,
        "get_storage_block_size": lambda spec: spec.block_size,
        "FullAttentionSpec": FullAttentionSpec,
        "is_c8_mxfp_kv_quant": lambda _config: False,
        "AscendSFAIndexerCacheSpec": AscendSFAIndexerCacheSpec,
        "get_single_raw_mla_backing": lambda raw: raw if isinstance(raw, torch.Tensor) else None,
        "logger": SimpleNamespace(info_once=lambda *args: logs.append(args)),
    }
    return _compile_function(_source_function("_reshape_kv_cache_v2"), scope), logs


def test_worker_uses_aligned_manager_page_and_selected_kernel_block(qfm, monkeypatch):
    monkeypatch.setitem(sys.modules, QFM_MODULE, qfm)
    manager_block_size = 512
    kernel_block_size = qfm.QFM_BLOCK
    kernel_pages_per_manager = manager_block_size // kernel_block_size
    qfm_payload = manager_block_size * qfm.QFM_WIDTH
    mamba_payload = qfm_payload + 1

    specs = {
        "attn": PageSpec(manager_block_size, qfm_payload, cache_qfm_c8=True),
    }
    mamba_specs = {
        "mamba": PageSpec(manager_block_size, mamba_payload),
    }
    aligned = _shared_mamba_alignment()(specs, mamba_specs, ["attn"])
    common_page = qfm.align_qfm_page_bytes(mamba_payload, [manager_block_size])

    assert common_page > mamba_payload
    assert common_page % (kernel_pages_per_manager * qfm.QFM_TILE) == 0
    assert aligned["attn"].block_size == manager_block_size
    assert aligned["attn"].page_size_padded == common_page
    assert aligned["mamba"].page_size_padded == common_page

    num_manager_pages = 3
    raw_offset = 64
    backing = torch.full((raw_offset + num_manager_pages * common_page + 64,), 0x5A, dtype=torch.uint8)
    raw = backing[raw_offset : raw_offset + num_manager_pages * common_page]
    group = SimpleNamespace(
        kv_cache_group_id=0,
        kv_cache_spec=aligned["attn"],
        layer_names=["attn"],
    )
    config = SimpleNamespace(layer_specs={"attn": aligned["attn"]}, num_blocks=num_manager_pages)
    reshape, logs = _reshape_qfm_branch()
    cache = reshape([group], {"attn": raw}, "auto", [kernel_block_size], {}, config)["attn"]

    assert cache.shape == (num_manager_pages * kernel_pages_per_manager, 1, 18, 128, 32)
    assert cache.stride() == (common_page // kernel_pages_per_manager, 128 * 576, 128 * 32, 32, 1)
    assert cache.storage_offset() == raw_offset
    assert logs and logs[0][1:4] == (manager_block_size, kernel_block_size, common_page)


def test_shared_mamba_alignment_without_qfm_is_unchanged():
    common_page = 294_913
    attention = PageSpec(512, common_page)
    mamba = PageSpec(512, common_page)
    specs = {"attn": attention}
    mamba_specs = {"mamba": mamba}

    aligned = _shared_mamba_alignment()(specs, mamba_specs, ["attn"])

    assert aligned["attn"] == attention
    assert aligned["mamba"] is mamba
    assert aligned["attn"].page_size_padded is None
    assert aligned["mamba"].page_size_padded is None

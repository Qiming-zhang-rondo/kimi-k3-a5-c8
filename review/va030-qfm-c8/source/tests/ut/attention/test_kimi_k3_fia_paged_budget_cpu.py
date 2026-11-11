# SPDX-License-Identifier: Apache-2.0
"""CPU accounting contracts for shared FP8/KDA plus private BF16 histories."""

import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@dataclass
class Uniform:
    kv_cache_specs: dict


def load_budget(specs, groups, override=None):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/worker.py"
    tree = ast.parse(path.read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_apply_fia_paged_memory_budget"
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    group_builder = Mock(return_value=groups)
    namespace = {"get_kv_cache_groups": group_builder, "UniformTypeKVCacheSpecs": Uniform, "logger": Mock()}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    worker = SimpleNamespace(
        get_kv_cache_spec=lambda: specs,
        cache_config=SimpleNamespace(num_gpu_blocks_override=override),
        vllm_config=object(),
    )
    return namespace["_apply_fia_paged_memory_budget"], worker, group_builder


def paged_spec(block_size=128, page_size=73728):
    return SimpleNamespace(
        block_size=block_size,
        page_size_bytes=page_size,
        cache_qfm_bf16_prefill=True,
        bf16_prefill_page_size_bytes=block_size * 576 * 2,
    )


@pytest.mark.parametrize("specs", [{}, None, {"ordinary": SimpleNamespace(page_size_bytes=73728)}])
def test_disabled_backend_keeps_existing_budget_and_override_semantics(specs):
    method, worker, group_builder = load_budget(specs, [], override=10)
    assert method(worker, 1000001) == 1000001
    group_builder.assert_not_called()


@pytest.mark.parametrize("memory", [0, 1001, 2**30, 2**60 + 7])
def test_pure_mla_reserves_twice_main_pool_for_bf16_without_float_rounding(memory):
    spec = paged_spec()
    specs = {"mla.0": spec, "mla.1": spec}
    groups = [SimpleNamespace(kv_cache_spec=spec, layer_names=list(specs))]
    method, worker, group_builder = load_budget(specs, groups)
    advertised = method(worker, memory)
    assert advertised == memory // 3
    blocks = advertised // (2 * spec.page_size_bytes)
    total = blocks * 2 * (spec.page_size_bytes + spec.bf16_prefill_page_size_bytes)
    assert total <= memory
    group_builder.assert_called_once_with(worker.vllm_config, specs)


@pytest.mark.parametrize("memory", [1000001, 2**30, 2**60 + 7])
def test_hybrid_budget_keeps_shared_base_pool_and_counts_only_opted_sidecars(memory):
    mla = paged_spec(block_size=768, page_size=884736)
    kda = SimpleNamespace(page_size_bytes=884736)
    draft = SimpleNamespace(page_size_bytes=884736, cache_qfm_bf16_prefill=False)
    specs = {"mla.0": mla, "mla.1": mla, "kda.0": kda, "kda.1": kda, "draft": draft}
    groups = [
        SimpleNamespace(kv_cache_spec=mla, layer_names=["mla.0", "mla.1"]),
        SimpleNamespace(
            kv_cache_spec=Uniform({"kda.0": kda, "kda.1": kda, "draft": draft}), layer_names=["kda.0", "kda.1", "draft"]
        ),
    ]
    method, worker, _ = load_budget(specs, groups)
    base_bytes = 3 * 884736  # Largest group, not the sum across overlaid groups.
    sidecar_bytes = 2 * 768 * 576 * 2
    advertised = method(worker, memory)
    assert advertised == memory * base_bytes // (base_bytes + sidecar_bytes)
    blocks = advertised // base_bytes
    assert blocks * (base_bytes + sidecar_bytes) <= memory


def test_group_promoted_manager_size_matches_actual_sidecar_allocation():
    original = paged_spec(block_size=128, page_size=73728)
    promoted = paged_spec(block_size=768, page_size=884736)
    groups = [SimpleNamespace(kv_cache_spec=promoted, layer_names=["mla"])]
    method, worker, _ = load_budget({"mla": original}, groups)
    memory = 2**30
    advertised = method(worker, memory)
    assert advertised == memory * promoted.page_size_bytes // (
        promoted.page_size_bytes + promoted.bf16_prefill_page_size_bytes
    )
    blocks = advertised // promoted.page_size_bytes
    assert blocks * (promoted.page_size_bytes + promoted.bf16_prefill_page_size_bytes) <= memory
    assert advertised < memory * promoted.page_size_bytes // (
        promoted.page_size_bytes + original.bf16_prefill_page_size_bytes
    )


@pytest.mark.parametrize("override", [1, 1000])
def test_forced_block_count_cannot_bypass_private_history_budget(override):
    spec = paged_spec()
    method, worker, group_builder = load_budget({"mla": spec}, [], override=override)
    with pytest.raises(ValueError, match="num_gpu_blocks_override"):
        method(worker, 2**30)
    group_builder.assert_not_called()


@pytest.mark.parametrize("base_page,extra_page", [(0, 147456), (73728, 0)])
def test_invalid_page_geometry_fails_before_allocator(base_page, extra_page):
    spec = paged_spec(page_size=base_page)
    spec.bf16_prefill_page_size_bytes = extra_page
    method, worker, _ = load_budget({"mla": spec}, [SimpleNamespace(kv_cache_spec=spec, layer_names=["mla"])])
    with pytest.raises(ValueError, match="positive"):
        method(worker, 2**30)


def test_both_profiled_and_explicit_memory_paths_apply_history_budget():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/worker.py"
    tree = ast.parse(path.read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "determine_available_memory"
    )
    calls = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_apply_fia_paged_memory_budget"
    ]
    assert len(calls) == 2
    manual = next(node for node in method.body if isinstance(node, ast.If))
    assert any(call in list(ast.walk(manual)) for call in calls)
    assert any(call not in list(ast.walk(manual)) for call in calls)

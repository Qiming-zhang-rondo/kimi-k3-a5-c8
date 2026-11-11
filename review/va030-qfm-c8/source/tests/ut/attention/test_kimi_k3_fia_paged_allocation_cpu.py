# SPDX-License-Identifier: Apache-2.0
"""Accounted independent BF16 histories beside shared hybrid FP8 pages.

These tests use CPU storage and existing native-writer stand-ins. They do not
validate FIA kernels, graph capture, RDMA completion or model performance.
"""

import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tests.ut.attention import test_kimi_k3_qfm_cpu as base
from tests.ut.attention import test_kimi_k3_qfm_pd_cpu as pd
from tests.ut.attention import test_kimi_k3_qfm_spec_copy_cpu as specs
from tests.ut.attention import test_kimi_k3_qfm_worker_blocks_cpu as worker

FP8_PAGE_BYTES = 128 * 576
BF16_PAGE_BYTES = FP8_PAGE_BYTES * 2


@pytest.fixture
def spec_type(monkeypatch):
    return specs.spec_type.__wrapped__(monkeypatch)


@pytest.fixture
def qfm(monkeypatch):
    return base.qfm.__wrapped__(monkeypatch)


def sidecar_spec(spec_type, manager=128, padding=0):
    return spec_type(
        block_size=manager,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.float8_e4m3fn,
        cache_qfm_c8=True,
        cache_qfm_bf16_prefill=True,
        page_size_padded=None if not padding else manager // 128 * (FP8_PAGE_BYTES + padding),
    )


@pytest.mark.parametrize("manager", [128, 256, 768])
@pytest.mark.parametrize("padding", [0, 32, 256])
def test_sidecar_budget_is_separate_from_shared_main_page(spec_type, manager, padding):
    original = sidecar_spec(spec_type, manager, padding)
    assert original.real_page_size_bytes == manager * 576
    assert original.bf16_prefill_page_size_bytes == manager * 576 * 2
    assert original.page_size_bytes == manager // 128 * (FP8_PAGE_BYTES + padding)
    copied = original.copy_with_new_block_size(128)
    assert copied.cache_qfm_bf16_prefill
    assert copied.real_page_size_bytes == FP8_PAGE_BYTES
    assert copied.bf16_prefill_page_size_bytes == BF16_PAGE_BYTES
    assert copied.page_size_bytes == FP8_PAGE_BYTES + padding
    assert original.block_size == manager
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=385),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )
    assert original.max_memory_usage_bytes(config) == (385 + manager - 1) // manager * original.page_size_bytes


def test_sidecar_is_a_distinct_merge_layout(spec_type):
    enabled = sidecar_spec(spec_type)
    assert spec_type.merge([enabled, enabled]).cache_qfm_bf16_prefill
    assert spec_type.merge([enabled, enabled]).real_page_size_bytes == FP8_PAGE_BYTES
    assert spec_type.merge([enabled, enabled]).bf16_prefill_page_size_bytes == BF16_PAGE_BYTES
    disabled = replace(enabled, cache_qfm_bf16_prefill=False)
    with pytest.raises(AssertionError, match="same Ascend KV cache layout"):
        spec_type.merge([enabled, disabled])
    assert disabled.page_size_bytes == FP8_PAGE_BYTES
    assert disabled.bf16_prefill_page_size_bytes == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"cache_qfm_c8": False},
        {"head_size": 512},
        {"dtype": torch.bfloat16},
        {"num_kv_heads": 2},
        {"scale_dim": 4},
        {"tokens_per_state": 2},
    ],
)
def test_sidecar_cannot_attach_to_other_cache_contracts(spec_type, changes):
    invalid = replace(sidecar_spec(spec_type), **changes)
    with pytest.raises(ValueError, match="full576 QFM"):
        _ = invalid.page_size_bytes


def reshape_one(qfm, monkeypatch, raw, spec, num_blocks=2):
    monkeypatch.setitem(sys.modules, worker.QFM_MODULE, qfm)
    group = SimpleNamespace(kv_cache_group_id=0, kv_cache_spec=spec, layer_names=["attn"])
    config = SimpleNamespace(layer_specs={"attn": spec}, num_blocks=num_blocks)
    reshape, logs = worker._reshape_qfm_branch()
    reshape.__globals__["get_dtype_size"] = lambda dtype: torch.empty(0, dtype=dtype).element_size()
    return reshape([group], {"attn": raw}, "auto", [128], {}, config)["attn"], logs


@pytest.mark.parametrize("manager", [128, 256, 768])
@pytest.mark.parametrize("padding", [0, 32, 256])
@pytest.mark.parametrize("offset", [0, 256])
def test_worker_preserves_main_stride_and_allocates_independent_history(
    spec_type, qfm, monkeypatch, manager, padding, offset
):
    spec = sidecar_spec(spec_type, manager, padding)
    physical = spec.page_size_bytes
    backing = torch.full((offset + 2 * physical + 256,), 0x5A, dtype=torch.uint8)
    raw = backing[offset : offset + 2 * physical]
    cache, logs = reshape_one(qfm, monkeypatch, raw, spec)
    assert cache.shape == (2 * manager // 128, 1, 18, 128, 32)
    assert cache.stride(0) == FP8_PAGE_BYTES + padding
    assert cache.storage_offset() == offset
    assert cache._qfm_bf16_prefill
    assert logs[0][1:4] == (manager, 128, physical)
    latent, auxiliary = cache._qfm_bf16_views
    assert latent.shape == (cache.shape[0], 1, 32, 128, 16)
    assert auxiliary.shape == (cache.shape[0], 1, 4, 128, 16)
    assert len({tensor.untyped_storage().data_ptr() for tensor in (cache, latent, auxiliary)}) == 3
    assert (latent.numel() + auxiliary.numel()) * 2 == 2 * spec.bf16_prefill_page_size_bytes
    for page in range(cache.shape[0]):
        latent[page].fill_(page + 0.5)
        auxiliary[page].fill_(-page - 0.5)
    slots = torch.tensor([0, 127, 128, cache.shape[0] * 128 - 1, -1])
    qfm.write_cache(torch.ones(5, 576), slots, cache, torch.ones(1))
    for page in range(cache.shape[0]):
        assert torch.all(latent[page] == page + 0.5)
        assert torch.all(auxiliary[page] == -page - 0.5)
        tail = offset + page * cache.stride(0) + FP8_PAGE_BYTES
        assert torch.all(backing[tail : offset + (page + 1) * cache.stride(0)] == 0x5A)
    assert latent.is_contiguous() and auxiliary.is_contiguous()
    assert torch.all(backing[:offset] == 0x5A)
    assert torch.all(backing[offset + 2 * physical :] == 0x5A)


def test_worker_rejects_sidecar_bytes_disagreeing_with_planner(qfm, monkeypatch):
    invalid = SimpleNamespace(
        block_size=128,
        head_size=576,
        dtype=torch.float8_e4m3fn,
        cache_qfm_c8=True,
        cache_qfm_bf16_prefill=True,
        page_size_padded=None,
        page_size_bytes=FP8_PAGE_BYTES,
        bf16_prefill_page_size_bytes=BF16_PAGE_BYTES + 32,
    )
    with pytest.raises(ValueError, match="reserved byte budget"):
        reshape_one(qfm, monkeypatch, torch.empty(2 * FP8_PAGE_BYTES, dtype=torch.uint8), invalid)


def hybrid_raw(spec_type, manager=128, padding=0, num_blocks=3):
    spec = sidecar_spec(spec_type, manager, padding)
    layer_bytes = num_blocks * spec.page_size_bytes

    class StateSpec:
        page_size_bytes = spec.page_size_bytes

    layer_specs = {"a": spec, "b": spec, "state": StateSpec()}
    common_backing_bytes = 2 * layer_bytes
    descriptors = [
        SimpleNamespace(
            size=common_backing_bytes,
            layers=["a", "b"],
            offset=0,
            layer_stride=layer_bytes,
            block_stride=spec.page_size_bytes,
        ),
        SimpleNamespace(
            size=common_backing_bytes,
            layers=["state"],
            offset=0,
            layer_stride=layer_bytes,
            block_stride=spec.page_size_bytes,
        ),
    ]
    config = SimpleNamespace(
        layer_specs=layer_specs,
        num_blocks=num_blocks,
        kv_cache_tensors=descriptors,
        kv_cache_groups=[SimpleNamespace(layer_names=list(layer_specs))],
    )
    attention_spec_type = next(cls for cls in spec_type.__mro__ if cls.__name__ == "AttentionSpec")
    scope = {
        "torch": torch,
        "get_current_vllm_config": lambda: SimpleNamespace(kv_transfer_config=None),
        "KVPPConfig": SimpleNamespace(from_vllm_config=lambda _: SimpleNamespace(size=1)),
        "_is_dsv4_model": lambda _: False,
        "_get_layer_kv_cache_specs": lambda value: value.layer_specs,
        "enable_sfa": lambda _: False,
        "can_share_full_and_sliding_kv_cache": lambda *_: False,
        "is_deepseek_v41_cache": lambda _: False,
        "MambaSpec": StateSpec,
        "AttentionSpec": attention_spec_type,
        "get_layerwise_reuse_config": lambda _: None,
        "get_kv_cache_tensor_layers": lambda descriptor: descriptor.layers,
        "AscendIndexerKPoolTailSpec": type("Tail", (), {}),
        "is_hidden_state_cache_spec": lambda _: False,
    }
    allocate = worker._compile_function(worker._source_function("_allocate_kv_cache"), scope)
    raw = allocate(config, {}, torch.device("cpu"))
    return raw, spec


@pytest.mark.parametrize("manager,padding", [(128, 0), (256, 256), (768, 512)])
def test_pd_publishes_only_fp8_payload_with_independent_history(qfm, manager, padding):
    pd.test_actual_registration_keeps_kernel_shape_offset_and_payload(qfm, manager, padding)


@pytest.mark.parametrize("manager,padding", [(128, 2 * FP8_PAGE_BYTES), (256, 256), (768, 256)])
def test_cross_group_distinct_block_ownership_is_preserved(spec_type, qfm, monkeypatch, manager, padding):
    # Attention owns block1 and KDA owns blocks0/2 in the same hybrid arena.
    # Dense-carving FP8 inside that arena would map Attention block1 into
    # KDA block0 and corrupt valid concurrent requests.
    raw, spec = hybrid_raw(spec_type, manager, padding)
    assert raw["a"].data_ptr() == raw["state"].data_ptr()
    assert raw["a"].is_contiguous() and raw["a"].numel() == 3 * spec.page_size_bytes
    assert raw["b"].storage_offset() - raw["a"].storage_offset() == raw["a"].numel()
    cache, _ = reshape_one(qfm, monkeypatch, raw["a"], spec, num_blocks=3)
    main_page_bytes = spec.page_size_bytes
    raw["state"][:main_page_bytes].fill_(42)
    raw["state"][2 * main_page_bytes :].fill_(43)
    protected_prefix = raw["state"][:main_page_bytes].clone()
    protected_suffix = raw["state"][2 * main_page_bytes :].clone()
    qfm.write_cache(torch.ones(1, 576), torch.tensor([manager]), cache, torch.ones(1))
    latent, auxiliary = cache._qfm_bf16_views
    ratio = manager // 128
    latent[ratio : 2 * ratio].fill_(11)
    auxiliary[ratio : 2 * ratio].fill_(12)
    assert torch.equal(raw["state"][:main_page_bytes], protected_prefix)
    assert torch.equal(raw["state"][2 * main_page_bytes :], protected_suffix)
    raw["state"][2 * main_page_bytes :].fill_(99)
    assert torch.all(latent[ratio : 2 * ratio] == 11)
    assert torch.all(auxiliary[ratio : 2 * ratio] == 12)
    assert torch.all(raw["b"] == 0)

# SPDX-License-Identifier: Apache-2.0
"""CPU wire/page tests; not Mooncake networking or model acceptance."""

import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.ut.attention import test_kimi_k3_qfm_cpu as base
from tests.ut.attention.test_kimi_k3_qfm_blocks_cpu import selection_inputs

qfm = base.qfm

ROOT = Path(__file__).resolve().parents[3]
MOONCAKE = ROOT / "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake"
spec = importlib.util.spec_from_file_location("qfm_pd_contract_test", MOONCAKE / "qfm_contract.py")
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)


def metadata(gap=0, manager=128, wire=contract.QFM_WIRE_FORMAT):
    return SimpleNamespace(
        cache_formats=[wire],
        block_shapes=[[(1, 18, 128, 32)]],
        block_lens=[[128 * 576]],
        block_strides=[[128 * 576 + gap]],
        block_size_scales=[[manager // 128]],
        layer_block_sizes=[manager],
    )


@pytest.mark.parametrize("manager", [128, 256, 512, 768])
def test_wire_allows_distinct_padding_and_manager_sizes(manager):
    contract.validate_qfm_transfer(metadata(256, manager), metadata(512, 128), 0, 0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("cache_formats", None),
        ("cache_formats", [""]),
        ("cache_formats", ["packed656"]),
        ("block_shapes", [[(128, 576)]]),
        ("block_lens", [[128 * 576 * 2]]),
        ("block_strides", [[128 * 576 + 1]]),
        ("block_size_scales", [[2]]),
    ],
)
def test_wire_rejects_wrong_or_old_endpoint(field, value):
    other = metadata()
    setattr(other, field, value)
    with pytest.raises(ValueError):
        contract.validate_qfm_transfer(metadata(), other, 0, 0)
    with pytest.raises(ValueError):
        contract.validate_qfm_transfer(other, metadata(), 0, 0)


def test_non_qfm_layers_unchanged():
    contract.validate_qfm_transfer(SimpleNamespace(), SimpleNamespace(), 0, 0)


@pytest.mark.parametrize("manager,gap", [(128, 0), (256, 256), (768, 512)])
def test_actual_registration_keeps_kernel_shape_offset_and_payload(qfm, manager, gap):
    tree = ast.parse((MOONCAKE / "base_worker.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MooncakeBaseConnectorWorker")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "register_kv_caches")
    mla_spec_type = type("MLA", (), {})
    ns = dict(
        torch=torch,
        math=math,
        MLAAttentionSpec=mla_spec_type,
        SlidingWindowMLASpec=type("SWA", (), {}),
        AscendSFAIndexerCacheSpec=type("Indexer", (), {}),
        QFM_WIRE_FORMAT=contract.QFM_WIRE_FORMAT,
        validate_qfm_transfer=contract.validate_qfm_transfer,
        get_kv_cache_tensor_layers=lambda _: ["target"],
        as_kv_cache_tensors=lambda c: (c,),
        collect_configured_register_regions=lambda *_: SimpleNamespace(ptrs=[], lengths=[]),
        validate_register_region_count=lambda _: None,
        global_te=SimpleNamespace(register_buffer=lambda *_: None),
        MooncakeTransferMetadata=SimpleNamespace,
        logger=SimpleNamespace(info=lambda *_: None, debug=lambda *_: None),
    )
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "registration", "exec"), ns)
    spec = mla_spec_type()
    spec.cache_qfm_c8, spec.block_size = True, manager
    pages = 2 * manager // 128
    raw = torch.zeros(256 + pages * (73728 + gap), dtype=torch.uint8)
    cache = qfm.cache_view(raw[256:], pages, 73728 + gap)
    worker = SimpleNamespace(
        kv_cache_config=SimpleNamespace(num_blocks=2, kv_cache_tensors=[None]),
        _build_kv_cache_spec_mappings=lambda: None,
        ascend_config=SimpleNamespace(kvpp_config=SimpleNamespace(size=1)),
        pcp_rank=0,
        tp_size=8,
        tp_rank=0,
        layer_name_to_group_index={"target": 0},
        layer_name_to_spec_index={"target": 0},
        kv_cache_specs=[spec],
        engine_id="P",
        te_rpc_port=1,
        block_size=manager,
        side_channel_host="localhost",
        handshake_port=2,
    )
    ns["register_kv_caches"](worker, {"target": cache})
    meta = worker.transfer_metadata
    assert meta.kv_caches_base_addr == [[cache.data_ptr()]]
    assert meta.block_shapes == [[(1, 18, 128, 32)]]
    assert meta.block_lens == [[73728]]
    assert meta.block_strides == [[73728 + gap]]
    assert meta.block_size_scales == [[manager // 128]]
    assert meta.cache_formats == [contract.QFM_WIRE_FORMAT]


@pytest.mark.parametrize("dcp", [1, 2, 4, 8])
def test_selection_requires_builtin_and_block_aligned_dcp(qfm, dcp):
    impl, config = selection_inputs(768)
    impl.num_heads = 12
    impl.vllm_config.parallel_config.decode_context_parallel_size = dcp
    impl.vllm_config.parallel_config.cp_kv_cache_interleave_size = 768
    impl.vllm_config.kv_transfer_config = SimpleNamespace(kv_connector="MooncakeConnectorV2")
    assert qfm.select_qfm(impl, config)
    if dcp > 1:
        impl.vllm_config.parallel_config.cp_kv_cache_interleave_size = 16
        with pytest.raises(ValueError, match="interleave"):
            qfm.select_qfm(impl, config)
    impl.vllm_config.kv_transfer_config.kv_connector = "MultiConnector"
    with pytest.raises(ValueError, match="built-in"):
        qfm.select_qfm(impl, config)


def community_mapper():
    """Execute the actual community block-mapping methods without NPU imports."""
    tree = ast.parse((MOONCAKE / "pull_worker.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MooncakePullRecvingThread")
    cls.bases = []
    names = {"_compute_group_block_ids", "_expand_block_ids", "_select_remote_tp_rank"}
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    full = type("FullAttentionSpec", (), {})
    ns = {
        "FullAttentionSpec": full,
        "KVCacheSpec": full,
        "AscendSFAIndexerCacheSpec": type("Indexer", (), {}),
        "SlidingWindowSpec": type("Sliding", (), {}),
        "MambaSpec": type("Mamba", (), {}),
        "is_circular_kv_cache_spec": lambda _: False,
    }
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "community_mapper", "exec"), ns)
    return ns[cls.name](), full()


@pytest.mark.parametrize("dcp", [1, 2, 4, 8])
@pytest.mark.parametrize("manager", [128, 256, 768])
@pytest.mark.parametrize("prefix_blocks", [0, 1])
def test_dcp_page_transfer_roundtrip(qfm, dcp, manager, prefix_blocks):
    mapper, spec = community_mapper()
    ratio = manager // 128
    # Two local manager blocks per rank; deliberately non-identical physical IDs.
    remote_ids = list(range((2 + prefix_blocks) * dcp - 1, -1, -1))
    producer_raw = torch.full((256 + len(remote_ids) * ratio * (73728 + 256),), 91, dtype=torch.uint8)
    producer = qfm.cache_view(producer_raw[256:], len(remote_ids) * ratio, 73728 + 256)
    for logical, physical in enumerate(remote_ids):
        for k in range(ratio):
            producer.view(torch.uint8)[physical * ratio + k].fill_((logical * ratio + k) % 251)
    for rank in range(dcp):
        mapper.dcp_size, mapper.dcp_rank = dcp, rank
        local_ids = [1, 0]
        mappings = mapper._compute_group_block_ids(
            "r",
            [[0]],
            1,
            0,
            manager,
            manager,
            local_ids,
            local_ids,
            remote_ids,
            len(remote_ids) * manager,
            len(remote_ids) * manager,
            prefix_blocks * dcp * manager,
            ratio,
            ratio,
            spec,
            0,
            dcp,
        )
        target_raw = torch.full((512 + 2 * ratio * (73728 + 512),), 77, dtype=torch.uint8)
        target = qfm.cache_view(target_raw[512:], 2 * ratio, 73728 + 512)
        for _, local_pages, remote_pages in mappings:
            for dst, src in zip(local_pages, remote_pages):
                target.view(torch.uint8)[dst].copy_(producer.view(torch.uint8)[src])
        for logical, physical in enumerate(local_ids):
            for k in range(ratio):
                expected = (((logical + prefix_blocks) * dcp + rank) * ratio + k) % 251
                assert (target.view(torch.uint8)[physical * ratio + k] == expected).all()
        assert (target_raw[:512] == 77).all()
        for page in range(2 * ratio):
            start = 512 + page * (73728 + 512) + 73728
            assert (target_raw[start : start + 512] == 77).all()


@pytest.mark.parametrize("local_dcp,remote_dcp", [(1, 2), (2, 4), (4, 2), (8, 1)])
@pytest.mark.parametrize("local_block,remote_block", [(768, 768), (128, 768), (768, 256)])
def test_asymmetric_dcp_maps_the_same_global_tokens(local_dcp, remote_dcp, local_block, remote_block):
    mapper, spec = community_mapper()
    total = 768 * 8 * 2
    local_ids = list(range(total // (local_dcp * local_block)))
    remote_ids = list(reversed(range(total // (remote_dcp * remote_block))))
    for rank in range(local_dcp):
        mapper.dcp_rank, mapper.dcp_size = rank, local_dcp
        mappings = mapper._compute_group_block_ids(
            "r",
            [list(range(remote_dcp))],
            remote_dcp,
            0,
            local_block,
            remote_block,
            local_ids,
            local_ids,
            remote_ids,
            total + 1,
            total,
            0,
            local_block // 128,
            remote_block // 128,
            spec,
            0,
            local_dcp,
        )
        seen = set()
        for remote_rank, local_pages, remote_pages in mappings:
            for lp, rp in zip(local_pages, remote_pages):
                local_id, lo = divmod(lp, local_block // 128)
                remote_id, ro = divmod(rp, remote_block // 128)
                local_token = (local_ids.index(local_id) * local_dcp + rank) * local_block + lo * 128
                remote_token = (remote_ids.index(remote_id) * remote_dcp + remote_rank) * remote_block + ro * 128
                assert local_token == remote_token
                assert lp not in seen
                seen.add(lp)
        assert len(seen) == total // local_dcp // 128

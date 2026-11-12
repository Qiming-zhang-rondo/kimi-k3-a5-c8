# SPDX-License-Identifier: Apache-2.0
"""QFM-only model wiring and independent physical-page COW contracts."""

import ast
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.ut.attention import test_kimi_k3_qfm_cpu as base
from tests.ut.attention import test_kimi_k3_qfm_worker_blocks_cpu as worker_blocks

qfm = base.qfm
ROOT = Path(__file__).resolve().parents[3]
MLA = ROOT / "vllm_ascend/attention/mla_v1.py"
PAYLOAD = 128 * 576


def compile_definition(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
    node = copy.deepcopy(node)
    node.decorator_list = []
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def compile_qfm_branch(method_name, parameters, namespace):
    tree = ast.parse(MLA.read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == method_name)
    branch = next(
        node
        for node in method.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Call)
        and isinstance(node.test.func, ast.Name)
        and node.test.func.id == "getattr"
        and isinstance(node.test.args[1], ast.Constant)
        and node.test.args[1].value == "use_qfm"
    )
    method = ast.FunctionDef(
        name="qfm_branch",
        args=ast.arguments(
            posonlyargs=[], args=[ast.arg(arg=name) for name in parameters], kwonlyargs=[], kw_defaults=[], defaults=[]
        ),
        body=[copy.deepcopy(branch)],
        decorator_list=[],
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(MLA), "exec"), namespace)
    return namespace["qfm_branch"]


@pytest.mark.parametrize("manager", [128, 256, 768])
def test_qfm_worker_view_retains_stride_and_explicit_page_axis_marker(qfm, monkeypatch, manager):
    monkeypatch.setitem(sys.modules, worker_blocks.QFM_MODULE, qfm)
    ratio = manager // 128
    page_bytes = ratio * (PAYLOAD + 256)
    spec = worker_blocks.PageSpec(manager, manager * 576, cache_qfm_c8=True, page_size_padded=page_bytes)
    backing = torch.full((256 + 2 * page_bytes + 256,), 0x5A, dtype=torch.uint8)
    raw = backing[256 : 256 + 2 * page_bytes]
    config = SimpleNamespace(layer_specs={"mla": spec}, num_blocks=2)
    group = SimpleNamespace(kv_cache_group_id=0, kv_cache_spec=spec, layer_names=["mla"])
    reshape, _ = worker_blocks._reshape_qfm_branch()
    result = reshape([group], {"mla": raw}, "auto", [128], {}, config)["mla"]
    assert result.shape == (2 * ratio, 1, 18, 128, 32)
    assert result.stride(0) == page_bytes // ratio
    assert result.storage_offset() == 256
    assert result._qfm_fp8_cache is True
    assert not hasattr(result, "_qfm_bf16_views")
    assert result.untyped_storage().data_ptr() == backing.data_ptr()
    assert torch.equal(backing, torch.full_like(backing, 0x5A))


@pytest.mark.parametrize("ratio", [1, 2, 6])
@pytest.mark.parametrize("managers", [2, 3])
@pytest.mark.parametrize("gap,offset", [(0, 0), (256, 256)])
def test_qfm_cow_copies_manager_kernel_pages_without_padding_or_alias_corruption(qfm, ratio, managers, gap, offset):
    stride = PAYLOAD + gap
    backing = torch.full((offset + managers * ratio * stride + 256,), 0x5A, dtype=torch.uint8)
    cache = qfm.cache_view(backing[offset : offset + managers * ratio * stride], managers * ratio, stride)
    cache._qfm_fp8_cache = True
    features = torch.arange(PAYLOAD)
    for kernel_page in range(managers * ratio):
        start = offset + kernel_page * stride
        backing[start : start + PAYLOAD] = ((features * 7 + kernel_page * 13) % 112).to(torch.uint8)
    before, expected = backing.clone(), backing.clone()
    copies = [SimpleNamespace(src_block_id=0, dst_block_id=1)]
    if managers == 3:
        copies.append(SimpleNamespace(src_block_id=1, dst_block_id=2))
    for operation in copies:
        for subpage in range(ratio):
            source = offset + (operation.src_block_id * ratio + subpage) * stride
            destination = offset + (operation.dst_block_id * ratio + subpage) * stride
            expected[destination : destination + PAYLOAD] = before[source : source + PAYLOAD]
    namespace = {"torch": torch}
    function = compile_definition(ROOT / "vllm_ascend/worker/utils.py", "copy_kv_cache_blocks_inplace", namespace)
    function([cache, cache], managers, copies)
    assert torch.equal(backing, expected)
    assert torch.equal(backing[offset : offset + PAYLOAD], before[offset : offset + PAYLOAD])


def test_legacy_fused_key_value_axis_remains_separate(qfm):
    cache = torch.arange(2 * 3 * 4 * 2 * 8).reshape(2, 3, 4, 2, 8).float()
    expected = cache.clone()
    expected[:, 1].copy_(expected[:, 0])
    function = compile_definition(
        ROOT / "vllm_ascend/worker/utils.py", "copy_kv_cache_blocks_inplace", {"torch": torch}
    )
    function([cache], 3, [SimpleNamespace(src_block_id=0, dst_block_id=1)])
    assert torch.equal(cache, expected)


@pytest.mark.parametrize("retain", [False, True])
def test_qfm_metadata_is_shared_without_legacy_prefill_split(retain):
    recorded = []
    slots = torch.tensor([7, -1, 9])
    batch = SimpleNamespace(slots=slots)
    common = SimpleNamespace(
        num_actual_tokens=2,
        num_input_tokens=3,
        num_reqs=1,
        query_start_loc=torch.tensor([0, 2]),
        seq_lens=torch.tensor([129]),
        _seq_lens_cpu=torch.tensor([129]),
        block_table_tensor=torch.tensor([[3, 0]]),
        attn_state=object(),
        causal=True,
    )
    owner = SimpleNamespace(
        qfm_state=SimpleNamespace(build=lambda value, **kwargs: recorded.append((value, kwargs)) or batch),
        metadata_cls=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    function = compile_definition(MLA, "_build_qfm", {})
    metadata = function(owner, common, retain_for_graph=retain)
    assert recorded == [(common, {"retain_for_graph": retain})]
    assert metadata.qfm is batch and metadata.slot_mapping is slots
    assert metadata.num_prefills == 0 and metadata.num_decode_tokens == 3
    assert metadata.block_tables is common.block_table_tensor


def test_qfm_forward_preserves_connector_order_and_single_cache_binding():
    events = []
    output = torch.empty(2, 4)
    cache = torch.zeros(2, 1, 18, 128, 32, dtype=torch.float8_e4m3fn)

    def execute(owner, hidden, active_cache, metadata, destination):
        assert active_cache is cache and destination is output
        events.append("attention")
        return destination

    namespace = {
        "torch": torch,
        "wait_for_kv_layer_from_connector": lambda name: events.append(("wait", name)),
        "forward_qfm": execute,
        "maybe_save_kv_layer_to_connector": lambda name, tensors: events.append(("save", name, tensors)),
    }
    function = compile_qfm_branch(
        "forward", ("self", "layer_name", "hidden_states", "kv_cache", "attn_metadata", "output"), namespace
    )
    result = function(SimpleNamespace(use_qfm=True), "mla", torch.ones(2, 4), cache, object(), output)
    assert result is output
    assert events[0] == ("wait", "mla") and events[1] == "attention"
    assert events[2][0:2] == ("save", "mla")
    assert len(events[2][2]) == 1 and events[2][2][0] is cache
    events.clear()
    with pytest.raises(RuntimeError, match="single"):
        function(SimpleNamespace(use_qfm=True), "mla", torch.ones(2, 4), (cache,), object(), output)
    assert events == []


def test_qfm_weight_reload_retains_scale_and_prolog_state(qfm, monkeypatch):
    weights = {"weights": torch.ones(1)}
    events = []

    class Prolog:
        def __init__(self, loaded, scale):
            assert loaded is weights
            self.scale = scale
            events.append("initialize")

        def update_weights(self, loaded):
            assert loaded is weights
            events.append("update")

    monkeypatch.setitem(sys.modules, "vllm_ascend.attention.kimi_k3_qfm_prolog", SimpleNamespace(QFMProlog=Prolog))
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.attention.kimi_k3_qfm_prolog_weights",
        SimpleNamespace(prepare_prolog_weights=lambda owner: weights),
    )
    namespace = {
        "torch": torch,
        "QFM_KV_DESCALE": 1.0,
        "load_ops": lambda: ("qfm", "metadata"),
        "logger": SimpleNamespace(info=lambda *args: None),
    }
    function = compile_qfm_branch("process_weights_after_loading", ("self",), namespace)
    owner = SimpleNamespace(use_qfm=True, W_UV=torch.ones(1), layer_name="mla", num_heads=12)
    function(owner)
    prolog, scale = owner.qfm_prolog, owner.qfm_scale
    scale.fill_(2)
    function(owner)
    assert events == ["initialize", "update"]
    assert owner.qfm_prolog is prolog and owner.qfm_scale is scale
    assert scale.tolist() == [1.0]
    assert owner.qfm_ops == ("qfm", "metadata")


def test_experimental_prefill_routes_are_absent_from_configuration_and_allocation():
    forbidden = {
        "enable_kimi_k3_qfm_fia_prefill",
        "kimi_k3_c8_prefill_backend",
        "kimi_k3_c8_fia_form",
        "enable_kimi_k3_mlapo_prefill",
        "cache_qfm_bf16_prefill",
        "bf16_prefill_page_size_bytes",
    }
    for relative in (
        "ascend_config.py",
        "attention/mla_v1.py",
        "core/kv_cache_interface.py",
        "worker/v2/attn_utils.py",
        "worker/utils.py",
    ):
        tree = ast.parse((ROOT / "vllm_ascend" / relative).read_text())
        names = {
            node.attr if isinstance(node, ast.Attribute) else node.id
            for node in ast.walk(tree)
            if isinstance(node, (ast.Attribute, ast.Name))
        }
        assert forbidden.isdisjoint(names)
        imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        assert all(not name or not any(part in name for part in ("kimi_k3_fia_", "kimi_k3_qfa_")) for name in imports)

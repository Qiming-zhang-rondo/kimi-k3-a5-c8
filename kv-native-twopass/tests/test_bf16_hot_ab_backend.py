"""CPU source/adapter contracts for the fixed BF16-hot A/B; not NPU proof."""

import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"hotkv"))
import kv_bf16_hot_ab as ab
import kv_native_merge as merge
from kv_window_backend import CompactBF16Plan
from kv_window_common import synthetic_case, validate_case, paged_array
from test_native_partition import TL, Pointer, allocate_outputs, oracle
from test_native_store_contract import Tensor as MetadataTensor, Kernel, fixture as store_fixture
from test_native_merge import Metadata, state_metadata, make_states, bf16
import test_native_merge as merge_tests
from test_native_twopass_backend import (Tensor, FakeBase, FakeScales, NativeTwoPassBackend,
                                        merge_torch, partition_numpy)
from test_native_hot_prepare_backend import FakePartition, fill_tensor, copy_tensor


ORDER = ("old", "recent_global", "recent_ring", "old_present", "recent_present", "invalid_rows")


def prepare_kernel_body(language):
    tree = ast.parse(Path(ab.__file__).read_text())
    function = next(x for x in ast.walk(tree) if isinstance(x, ast.FunctionDef)
                    and x.name == "_bf16_hot_prepare_kernel")
    function.decorator_list = []
    scope = {"tl": language}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 ab.__file__, "exec"), scope)
    return scope[function.name]


class ExactBf16PrepareTests(unittest.TestCase):
    def execute(self, ids, positions, slots, *, window=8, capacity=16, width=8, seed=17,
                outputs=None, latent=None, rope=None):
        rows, _, sparse = ids.shape
        rng = np.random.default_rng(seed)
        current = rng.integers(0, 256, (rows, 1024), np.uint8)
        current_rope = rng.integers(0, 256, (rows, 128), np.uint8)
        outputs = allocate_outputs(rows, sparse, width, True) if outputs is None else outputs
        latent = np.full((48, 1024), 165, np.uint8) if latent is None else latent
        rope = np.full((48, 128), 165, np.uint8) if rope is None else rope
        expected_latent, expected_rope = latent.copy(), rope.copy()
        for row, slot in enumerate(slots):
            if 0 <= slot < 48:
                expected_latent[slot], expected_rope[slot] = current[row], current_rope[row]
        arrays = [ids, positions, *(outputs[name] for name in ORDER), current, current_rope, slots, latent, rope]
        names = ["indices", "positions", *ORDER, "current", "current_rope", "slots", "latent", "rope"]
        pointers = [Pointer(value, name) for value, name in zip(arrays, names)]
        language = TL()
        kernel = prepare_kernel_body(language)
        for row in range(rows):
            language.row = row
            kernel(*pointers, S=sparse, HW=width, WINDOW=window, CAP=capacity,
                   BLOCK=1 << (sparse-1).bit_length(), KV_BYTES=1024, ROPE_BYTES=128,
                   HOT_CAPACITY=48, STORE_BLOCK=1024)
        expected = oracle(ids, positions, window, capacity, width, True)
        counts = np.sum((ids[:, 0] >= 0) & (ids[:, 0] <= positions[:, None]) &
                        (ids[:, 0] > positions[:, None]-window), axis=1)
        expected["invalid_rows"] |= (counts != width) | (slots < 0) | (slots >= 48)
        for name in ORDER:
            np.testing.assert_array_equal(outputs[name], expected[name], err_msg=name)
            np.testing.assert_array_equal(language.writes[name], np.ones(outputs[name].size, np.int32))
        for name, actual, wanted in (("latent", latent, expected_latent), ("rope", rope, expected_rope)):
            np.testing.assert_array_equal(actual, wanted, err_msg=name)
            writes = np.zeros_like(actual, np.int32)
            writes[slots[(slots >= 0) & (slots < 48)]] = 1
            np.testing.assert_array_equal(language.writes[name], writes.reshape(-1), err_msg=name)
        self.assertEqual(language.scans, rows*2)
        return outputs, latent, rope

    def test_full_hot_unsorted_prefix_ring_wrap_and_all_bytes_exact(self):
        ids = np.array([[[31, 1, 35, 30, 32, 2, 33, 34, 0, 29, 28] + [-1]*6]], np.int32)
        out, latent, rope = self.execute(ids, np.array([35], np.int64), np.array([47], np.int64))
        self.assertFalse(out["invalid_rows"].any())
        self.execute(ids[:, :, ::-1].copy(), np.array([35], np.int64), np.array([0], np.int32),
                     outputs=out, latent=latent, rope=rope, seed=98)

    def test_missing_hot_future_slots_and_empty_rows_flagged_without_oob(self):
        ids = np.array([[31, 35, 32] + [-1]*14,
                        [2, -2, 99] + [-1]*14,
                        [-1]*17,
                        [1, 2] + [-1]*15], np.int32)[:, None, :]
        outputs, _, _ = self.execute(ids, np.array([35, 35, 35, -1], np.int64),
                                      np.array([47, -1, 48, 31], np.int64))
        self.assertTrue(outputs["invalid_rows"].all())


def metadata_fixture():
    fixture = store_fixture()
    fixture.base.torch.bool = "bool"
    tensor = lambda shape, dtype: MetadataTensor(shape, dtype, events=fixture.events)
    fixture.hot = tensor((1, 16, 1, 512), "bf16")
    fixture.hot_rope = tensor((1, 16, 1, 64), "bf16")
    layout = SimpleNamespace(rows=3, sparse_capacity=17, hot_width=8, window=8, ring_capacity=16,
                             remap_ring=True, block=32, variant="two_scan")
    outputs = {name: tensor((3, 1, 17 if name == "old" else 8), "int32") if i < 3
               else tensor((3,), "bool") for i, name in enumerate(ORDER)}
    fixture.partition = SimpleNamespace(layout=layout, outputs=outputs,
        _inputs=(tensor((3, 1, 17), "int32"), tensor((3,), "int64"), *(outputs[name] for name in ORDER)),
        metadata=dict(variant="two_scan"), call=mock.Mock(side_effect=AssertionError("standalone partition")))
    return fixture


class PrepareContractTests(unittest.TestCase):
    def prepare(self, fixture):
        kernel = Kernel()
        with mock.patch.object(ab, "_load_prepare_kernel", return_value=kernel):
            prepared = ab.PreparedBf16HotStorePartition(fixture.base, fixture.hot, fixture.hot_rope,
                                                       fixture.slots, fixture.partition)
        return prepared, kernel

    def test_one_launch_no_quantizer_no_timed_host_access_and_original_bf16_storage(self):
        data = metadata_fixture()
        prepared, kernel = self.prepare(data)
        data.events.clear()
        for _ in range(2):
            self.assertIs(prepared.call(), data.partition.outputs)
        self.assertFalse({"detach", "cpu", "numpy"} & set(data.events))
        data.partition.call.assert_not_called()
        data.base.quantize.assert_not_called()
        data.base.torch_npu.npu_dynamic_mx_quant.assert_not_called()
        data.base.torch_npu.npu_dynamic_block_quant.assert_not_called()
        self.assertEqual(len(kernel.calls), 2)
        for grid, args, constants in kernel.calls:
            self.assertEqual(grid, (3,))
            self.assertTrue(all(a is b for a, b in zip(args[:8], data.partition._inputs)))
            for pointer, source in ((args[8], data.base.current_kv), (args[9], data.base.current_rope),
                                    (args[11], data.hot), (args[12], data.hot_rope)):
                self.assertEqual(pointer.dtype, "uint8")
                self.assertIs(pointer.storage, source.storage)
            self.assertEqual(constants["KV_BYTES"], 1024)
            self.assertEqual(constants["ROPE_BYTES"], 128)

    def test_reject_wrong_shapes_format_mapping_slots_and_device(self):
        for name in ("dtype", "dim", "rope_shape", "slots_duplicate", "slots_range", "slots_dtype",
                     "buffer_identity", "partition_dtype", "device", "contiguous", "rows", "variant"):
            with self.subTest(name=name):
                data = metadata_fixture()
                if name == "dtype": data.hot.dtype = "fp32"
                elif name == "dim": data.base.dim = 256
                elif name == "rope_shape": data.hot_rope.shape = (1, 8, 1, 64)
                elif name == "slots_duplicate": data.slots.values = np.array([1, 1, 8])
                elif name == "slots_range": data.slots.values = np.array([1, 16, 8])
                elif name == "slots_dtype": data.slots.dtype = "fp32"
                elif name == "buffer_identity": data.partition.outputs["old"] = object()
                elif name == "partition_dtype": data.partition._inputs[0].dtype = "int64"
                elif name == "device": data.hot_rope.device = "npu:1"
                elif name == "contiguous": data.hot_rope.contiguous = False
                elif name == "rows": data.partition.layout.rows = 4
                else: data.partition.layout.variant = "packed_one_scan"
                with self.assertRaises(ValueError):
                    self.prepare(data)


class MixedMergeTests(unittest.TestCase):
    def test_same_kernel_original_heterogeneous_pointers_and_fp32_destination(self):
        left, right = state_metadata(dtype="torch.float32"), state_metadata(dtype="torch.bfloat16")
        out, presence = Metadata((3, 2, 527), "torch.float32"), Metadata((3,), "torch.bool")
        kernel = Kernel()
        with mock.patch.object(merge, "_load_kernel", return_value=kernel):
            self.assertIs(ab.merge_mixed_dtype(left, right, out, left_present=presence, right_present=presence), out)
        grid, args, constants = kernel.calls[0]
        self.assertEqual(grid, (3, 6))
        self.assertEqual(args, (*left, *right, presence, presence, out))
        self.assertEqual(constants, dict(H=2, D=527, BLOCK=256, HAS_LEFT_PRESENT=True,
                                         HAS_RIGHT_PRESENT=True, DIRECT=False))
        self.assertEqual(out.dtype, "torch.float32")

    def test_independent_dtype_validation_still_rejects_shape_device_stats_and_downcast(self):
        for failure in ("shape", "device", "stats", "output", "hot_dtype"):
            with self.subTest(failure=failure):
                left, right = state_metadata(dtype="torch.float32"), state_metadata(dtype="torch.bfloat16")
                out = Metadata((3, 2, 527), "torch.float32")
                if failure == "shape": right = state_metadata((4, 2, 527))
                elif failure == "device":
                    for value in right: value.device = "npu:1"
                elif failure == "stats": right[1].dtype = "torch.bfloat16"
                elif failure == "output": out.dtype = "torch.bfloat16"
                else: right[0].dtype = "torch.float16"
                with self.assertRaises(ValueError):
                    ab.merge_mixed_dtype(left, right, out)

    def test_source_kernel_retains_cold_fp32_precision_and_normalizes_empty_sides(self):
        left, right = make_states((4, 2, 512), dtype="fp32")
        right[0] = bf16(right[0])
        lp, rp = np.array([True, True, False, False]), np.array([True, False, True, False])
        output = merge_tests.TestActualKernelBody().execute(left, right, lp, rp, output_dtype="fp32")
        np.testing.assert_array_equal(output[1], left[0][1])
        np.testing.assert_array_equal(output[2], right[0][2])
        np.testing.assert_array_equal(output[3], 0)
        self.assertTrue(np.any(output[1] != bf16(output[1])))
        lm, ls, rm, rs = [np.asarray(x, np.float64) for x in (left[1], left[2], right[1], right[2])]
        maximum = np.maximum(lm, rm)
        lw, rw = ls*np.exp(lm-maximum), rs*np.exp(rm-maximum)
        expected = (left[0]*lw+right[0]*rw)/(lw+rw)
        np.testing.assert_allclose(output[0], expected[0], rtol=1e-5, atol=1e-6)

    def test_merge_oracle_rounds_only_to_native_destination_contract(self):
        values = np.random.default_rng(87).normal(size=(4, 64, 512))
        rounded = ab.merge_reference_for_dtype(values, "torch.bfloat16")
        error = np.linalg.norm(rounded-values)/np.linalg.norm(values)
        self.assertGreater(error, .001)  # Correct BF16 output would fail an unrounded .001 gate.
        np.testing.assert_array_equal(rounded, bf16(values))
        np.testing.assert_array_equal(ab.merge_reference_for_dtype(values, "torch.float32"), values.astype(np.float32))
        with self.assertRaises(ValueError):
            ab.merge_reference_for_dtype(values, "torch.float16")


class Bf16FakeBase(FakeBase):
    def __init__(self, data):
        super().__init__(data)
        self.rope_dim = data["q_rope"].shape[-1]
        self.bf16_indices = "compact"
        self.current_rope = Tensor(np.stack([data["k_rope"][b, p]
                                            for (b, _), p in zip(self.mapping, self.positions)]))
        self.global_rope = paged_array(data["k_rope"], self.page, self.table_cpu)

    def make_bf16_plan(self, indices, counts=None, label="full"):
        return CompactBF16Plan(self, indices, counts, label)

    def _read(self, kind, indices, *, cache, rope=None, table=None, lengths=None, statistics, bf16_plan=None):
        self.reads.append(dict(kind=kind, indices=indices, cache=cache, rope=rope, table=table,
                               lengths=lengths, statistics=statistics, bf16_plan=bf16_plan))
        table = self.table if table is None else table
        if kind == "bf16":
            assert bf16_plan.uniform_count == indices.shape[-1]
            assert np.all(np.asarray(indices) >= 0)
            values, ropes = np.asarray(cache), np.asarray(rope)
        else:
            values = np.asarray(cache).view(np.uint8).view(np.float32).reshape(-1, self.page, 1, self.dim)
            ropes = self.global_rope
        output = np.empty((self.rows, self.heads, self.dim), np.float32)
        maximum = np.empty((self.rows, self.heads, 1), np.float32)
        total = np.empty_like(maximum)
        for row, (b, _) in enumerate(self.mapping):
            ids = np.asarray(indices)[row, 0]
            ids = ids[ids >= 0]
            blocks, offsets = np.asarray(table)[b, ids//self.page], ids % self.page
            kv = values[blocks, offsets, 0].astype(np.float64)
            kr = ropes[blocks, offsets, 0].astype(np.float64)
            score = (np.asarray(self.q_nope[row], np.float64) @ kv.T +
                     np.asarray(self.q_rope[row], np.float64) @ kr.T)*self.scale
            maximum[row] = score.max(-1, keepdims=True)
            exp = np.exp(score-maximum[row])
            total[row] = exp.sum(-1, keepdims=True)
            output[row] = exp @ kv/total[row]
        return Tensor(output), Tensor(maximum), Tensor(total)


class FakeBf16Preparation:
    def __init__(self, base, cache, rope, slots, partition, events):
        self.base, self.cache, self.rope, self.slots = base, cache, rope, slots
        self.partition, self.outputs, self.events = partition, partition.outputs, events
        self.calls = 0
        self.skip_latent = self.skip_rope = self.skip_partition = False
        self.metadata = dict(implementation="CPU_BF16_PREPARE_DOUBLE", quantizer_api_calls=0)

    def call(self):
        self.calls += 1
        self.events.append("bf16_prepare")
        if not self.skip_latent:
            self.cache.view(-1, 1, self.base.dim).index_copy_(0, self.slots, self.base.current_kv)
        if not self.skip_rope:
            self.rope.view(-1, 1, self.base.rope_dim).index_copy_(0, self.slots, self.base.current_rope)
        if not self.skip_partition:
            self.partition.produce()
        return self.outputs


class Bf16BackendTests(unittest.TestCase):
    def setUp(self):
        # Small CPU adapter fixture; the production fixed-shape guard is tested
        # independently below. No fake numerical format is an A5 assertion.
        data = synthetic_case(batch=1, queries=4, keys=257, heads=2, dim=64, rope_dim=64,
                              topk=16, page=128)
        data["indices"][:] = -1
        for query in range(4):
            position = 257-4+query
            data["indices"][0, query, 0, :4] = [position, 0, position-1, 1]
        self.data = validate_case(data)
        scales = FakeScales(self.data)
        scales.base = Bf16FakeBase(self.data)
        self.backend = NativeTwoPassBackend(self.data, scale_backend=scales)
        self.base, self.events = scales.base, []

        def part(*args, **kwargs):
            return FakePartition(*args, **kwargs, events=self.events)

        def prepare(*args):
            return FakeBf16Preparation(*args, events=self.events)

        def merge_cpu(left, right, out, *, left_present=None, right_present=None):
            self.events.append("merge")
            out[...] = merge_torch(self.base.torch, left, right, left_present, right_present)
            return out

        for patch in (mock.patch.object(ab, "validate_fixed_case"),
                      mock.patch.object(ab, "PreparedBf16HotStorePartition", side_effect=prepare),
                      mock.patch.object(ab, "merge_mixed_dtype", side_effect=merge_cpu),
                      mock.patch.dict(sys.modules, {"kv_native_partition": SimpleNamespace(PreparedPartition=part),
                                                    "kv_native_merge": SimpleNamespace(merge_states=merge_cpu)}),
                      mock.patch.object(Tensor, "fill_", fill_tensor, create=True),
                      mock.patch.object(Tensor, "copy_", copy_tensor, create=True)):
            patch.start()
            self.addCleanup(patch.stop)

    def prepare(self, writer="native"):
        return ab.Bf16HotRunner(self.backend, window=2, writer=writer)

    def test_native_bf16_writer_and_reader_keep_original_rope_no_hot_quantizer(self):
        runner = self.prepare()
        self.base.reads.clear()
        self.base.writes.clear()
        self.events.clear()
        expected = runner.call().copy()
        actual = runner.run(write=True, dynamic=True)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        self.assertEqual([x["kind"] for x in self.base.reads], ["c4", "bf16", "c4", "bf16"])
        self.assertEqual([x[0] for x in self.base.writes], ["baseline"])
        hot = self.base.reads[-1]
        self.assertIs(hot["rope"], runner.hot_rope)
        self.assertIs(hot["bf16_plan"], runner.bf16_plan)
        self.assertEqual(runner.metadata["bf16_hot_read_plan"]["operator_calls"], 1)
        self.assertEqual(runner.metadata["bf16_hot_read_plan"]["valid_counts"], [2]*4)
        self.assertEqual(runner.metadata["merge_dtype_cast_kernel_calls"], 0)
        self.assertEqual(self.events, ["merge", "standalone_partition", "merge"])
        np.testing.assert_array_equal(runner.decoded_hot_kv(), self.data["kv"])
        self.assertFalse(np.array_equal(runner.decoded_hot_kv(), self.base.decoded_kv("c8")))

    def test_fused_write_dynamic_has_one_prepare_no_duplicate_partition(self):
        runner = self.prepare("hot_store_partition")
        self.events.clear()
        self.base.writes.clear()
        self.base.reads.clear()
        with mock.patch.object(runner, "partition", side_effect=AssertionError("duplicate partition")):
            runner.run(write=True, dynamic=True)
        self.assertEqual(self.events, ["bf16_prepare", "merge"])
        self.assertEqual(runner._hot_prepare.calls, 1)
        self.assertEqual(runner._partition_runner.calls, 0)
        self.assertEqual([x[0] for x in self.base.writes], ["baseline"])
        self.assertEqual([x["kind"] for x in self.base.reads], ["c4", "bf16"])
        self.assertIs(self.base.reads[0]["indices"], runner._hot_prepare.outputs["old"])
        self.assertIs(self.base.reads[1]["indices"], runner._hot_prepare.outputs["recent_ring"])
        stages = runner.component_calls()
        self.assertEqual(set(stages), {"cold_write", "hot_prepare", "cold_read", "hot_read", "merge"})
        self.assertEqual(stages["hot_prepare"]().dtype, np.uint8)

    def test_poison_gates_detect_missing_rope_or_latent_and_restore_every_buffer(self):
        runner = self.prepare("hot_store_partition")
        saved = dict(cold=runner.cold_cache.copy(), hot=runner.hot_cache.copy(), rope=runner.hot_rope.copy(),
                     **{name: value.copy() for name, value in runner._hot_prepare.outputs.items()})
        def restored():
            for name, value in (("cold", runner.cold_cache), ("hot", runner.hot_cache), ("rope", runner.hot_rope),
                                *runner._hot_prepare.outputs.items()):
                np.testing.assert_array_equal(value, saved[name], err_msg=name)
        self.assertTrue(runner.writer_diagnostics()["passed"])
        restored()
        for field, switch in (("hot_latent", "skip_latent"), ("hot_rope", "skip_rope")):
            setattr(runner._hot_prepare, switch, True)
            report = runner.writer_diagnostics()
            self.assertFalse(report["passed"])
            self.assertFalse(report["fields"][field]["passed"])
            restored()
            setattr(runner._hot_prepare, switch, False)
        self.assertTrue(runner.partition_diagnostics()["passed"])
        restored()
        runner._hot_prepare.skip_partition = True
        self.assertFalse(runner.partition_diagnostics()["passed"])
        restored()

    def test_native_writer_gate_uses_independent_expected_bytes(self):
        runner = self.prepare()
        self.assertTrue(runner.writer_diagnostics()["passed"])
        with mock.patch.object(runner, "_hot_write", return_value=runner.hot_cache):
            report = runner.writer_diagnostics()
        self.assertFalse(report["passed"])
        self.assertFalse(report["fields"]["hot_latent"]["passed"])
        self.assertFalse(report["fields"]["hot_rope"]["passed"])

    def test_reader_has_independent_hot_reference_stats_merge_and_zero_query_gates(self):
        runner = self.prepare()
        report = runner.reader_diagnostics()
        self.assertTrue(report["passed"], report)
        self.assertEqual(set(report["checks"]), {"hot_output_vs_own_reference", "hot_lse_vs_own_reference",
                                                "mixed_dtype_merge_vs_host_native_states"})
        self.assertTrue(runner.zero_query_diagnostics()["passed"])
        actual_read = runner._hot_read
        def wrong_read(*args, **kwargs):
            output, maximum, total = actual_read(*args, **kwargs)
            return output*2, maximum, total
        with mock.patch.object(runner, "_hot_read", side_effect=wrong_read):
            self.assertFalse(runner.reader_diagnostics()["passed"])

    def test_reject_changed_hot_count_and_unsupported_recipe_writer(self):
        self.backend.data["indices"][0, 0, 0, 2] = 2
        with self.assertRaisesRegex(ValueError, "exactly W"):
            self.prepare()
        for kwargs in (dict(writer="fused_store"), dict(recipe="invented")):
            with self.assertRaises(ValueError):
                ab.Bf16HotRunner(self.backend, **kwargs)


class FixedScopeTests(unittest.TestCase):
    def test_guard_rejects_other_shape_lengths_and_padded_policy(self):
        def base():
            return SimpleNamespace(batch=1, queries=4, keys=8192, heads=64, dim=512, rope_dim=64,
                                   indices=SimpleNamespace(shape=(4, 1, 2048)), page=128, bf16_indices="compact",
                                   data={"q_lengths": np.array([4]), "k_lengths": np.array([8192])})
        ab.validate_fixed_case(base(), 128)
        for name in ("batch", "queries", "keys", "heads", "dim", "rope_dim", "page", "counts", "policy", "window"):
            value, window = base(), 128
            if name == "counts": value.data["q_lengths"] = np.array([3])
            elif name == "policy": value.bf16_indices = "padded"
            elif name == "window": window = 256
            else: setattr(value, name, getattr(value, name)+1)
            with self.subTest(name=name), self.assertRaises(ValueError):
                ab.validate_fixed_case(value, window)


if __name__ == "__main__":
    unittest.main()

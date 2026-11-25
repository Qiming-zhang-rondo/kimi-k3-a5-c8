"""Pure CPU tests: no Torch, CANN, device launches or serving claims."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfa_recent128.contract import (
    ContractError, Recent128Config, fusion_eligibility, parse_recent128_config,
    partition_reference, plan_step, validate_runtime_contract,
)


def tags_through(end, capacity):
    tags = np.full(capacity, -1, np.int64)
    positions = np.arange(max(0, end-capacity), end, dtype=np.int64)
    tags[positions % capacity] = positions
    return tags


def config(**kwargs):
    return Recent128Config(diagnostics_path="/tmp/recent128-contract-test.json", **kwargs)


class Recent128ContractTest(unittest.TestCase):
    def test_preprocess_default_preserved_and_explicit_off_on_native_accuracy(self):
        self.assertEqual(parse_recent128_config().preprocess, "preserve")
        off = parse_recent128_config({"sfa_recent128": dict(window=0, preprocess="native_accuracy")})
        on = parse_recent128_config({"sfa_recent128": dict(window=128, preprocess="native_accuracy",
            diagnostics_path="/tmp/recent128-contract-test.json")})
        self.assertFalse(off.enabled)
        self.assertTrue(on.enabled)
        self.assertEqual(off.preprocess, on.preprocess)
        self.assertEqual(on.execution, "accuracy_eager")
        for bad in ("native", "NATIVE", "prolog", None, True):
            with self.subTest(preprocess=bad), self.assertRaisesRegex(ContractError, "preprocess"):
                Recent128Config(preprocess=bad)
        with self.assertRaisesRegex(ContractError, "accuracy_eager"):
            Recent128Config(preprocess="native_accuracy", execution="serving")

    def test_config_disabled_and_cold_quantization_cannot_be_changed(self):
        disabled = parse_recent128_config({"other_feature": {"foo": 1}})
        self.assertFalse(disabled.enabled)
        self.assertEqual(disabled.capacity, 0)
        # Disabled fast paths never inspect unsupported runtime/lifecycle state.
        validate_runtime_contract(disabled)
        step = plan_step(disabled, seq_len=None, query_count=None, previous_end=object())
        self.assertFalse(step.enabled)
        self.assertIsNone(step.positions)
        with self.assertRaisesRegex(ContractError, "baseline quantization"):
            parse_recent128_config({"sfa_recent128": {"window": 128, "scale_alg": 2}})
        for bad in (True, -1, 64, 256):
            with self.subTest(window=bad), self.assertRaises(ContractError):
                Recent128Config(window=bad)
        with self.assertRaises(ContractError):
            Recent128Config(window=128, execution="serving")
        with self.assertRaisesRegex(ContractError, "diagnostics_path"):
            Recent128Config(window=128)
        with self.assertRaisesRegex(ContractError, "<=1024"):
            config(window=128, max_query_chunk=1025)
        with self.assertRaisesRegex(ContractError, "source_root"):
            config(window=128, reader="fused_eligible", library_dir="/tmp/lib")

    def test_runtime_guards_do_not_change_preprocessing(self):
        cfg = config(window=128)
        valid = dict(preprocess_type="NATIVE", max_num_seqs=1, enforce_eager=True)
        validate_runtime_contract(cfg, **valid)
        for key, value in (("preprocess_type", "PROLOG_V3"), ("max_num_seqs", 2),
                           ("enforce_eager", False), ("capturing", True),
                           ("prefix_caching", True), ("pd_enabled", True),
                           ("pcp_size", 2), ("dcp_size", 2), ("speculative_enabled", True)):
            args = dict(valid, **{key: value})
            with self.subTest(key=key), self.assertRaises(ContractError):
                validate_runtime_contract(cfg, **args)
        self.assertEqual(valid["preprocess_type"], "NATIVE")

    def test_step_reset_contiguous_prefill_chunks_and_rejected_rollback(self):
        cfg = config(window=128, max_query_chunk=32)
        self.assertEqual(cfg.capacity, 256)
        first = plan_step(cfg, seq_len=32, query_count=32, previous_end=None)
        self.assertTrue(first.reset)
        np.testing.assert_array_equal(first.positions, np.arange(32))
        following = plan_step(cfg, seq_len=64, query_count=32, previous_end=first.end)
        self.assertFalse(following.reset)
        self.assertEqual((following.start, following.end), (32, 64))
        # A new request at position0 resets all per-layer tag state.
        self.assertTrue(plan_step(cfg, seq_len=3, query_count=3, previous_end=64).reset)
        for end, count, previous in ((64, 32, None), (64, 32, 31), (64, 32, 33),
                                      (65, 33, 32), (1, 2, None), (0, 0, 0)):
            with self.subTest(end=end, count=count), self.assertRaises(ContractError):
                plan_step(cfg, seq_len=end, query_count=count, previous_end=previous)

    def test_capacity_preserves_first_query_window_after_whole_chunk_write(self):
        for chunk in (1, 32, 128, 1024):
            cfg = config(window=128, max_query_chunk=chunk)
            self.assertGreaterEqual(cfg.capacity, 128+chunk-1)
            self.assertEqual(cfg.capacity % 128, 0)
            start = 1500
            step = plan_step(cfg, seq_len=start+chunk, query_count=chunk, previous_end=start)
            tags = tags_through(step.end, cfg.capacity)
            first_window = np.arange(start-127, start+1, dtype=np.int64)
            result = partition_reference(first_window[None, :], positions=[start], tags=tags)
            np.testing.assert_array_equal(result["recent_global"][0,0], first_window)
            self.assertEqual(result["recent_counts"].tolist(), [128])

    def test_stable_partition_wrap_minus_one_tail_causal_and_unchanged_topk(self):
        tags = tags_through(301, 256)
        topk = np.array([[[3, 299, 173, 172, 300, 999, -1, -1]]], np.int32)
        before = topk.copy()
        result = partition_reference(topk, positions=[300], tags=tags)
        np.testing.assert_array_equal(topk, before)
        self.assertEqual(result["old"][0,0,:2].tolist(), [3, 172])
        self.assertEqual(result["recent_global"][0,0,:3].tolist(), [299, 173, 300])
        self.assertEqual(result["recent_ring"][0,0,:3].tolist(), [43, 173, 44])
        self.assertEqual(result["old_counts"].tolist(), [2])
        self.assertEqual(result["recent_counts"].tolist(), [3])
        self.assertTrue(result["old_present"][0] and result["recent_present"][0])
        self.assertTrue(np.all(result["old"][0,0,2:] == -1))

    def test_missing_original_bf16_tag_raises_not_c4_dequant_fallback(self):
        tags = tags_through(301, 256)
        tags[299 % 256] = 43  # A stale logical token in the same physical ring slot.
        with self.assertRaisesRegex(ContractError, "original BF16"):
            partition_reference([[299]], positions=[300], tags=tags)
        with self.assertRaisesRegex(ContractError, "original BF16"):
            partition_reference([[299]], positions=[300], tags=None, capacity=256)

    def test_empty_allhot_and_window0_do_not_invent_tokens_or_read_tags(self):
        empty = partition_reference([[-1, -1]], positions=[100], tags=tags_through(101, 256))
        self.assertFalse(empty["old_present"][0] or empty["recent_present"][0])
        self.assertEqual(empty["old_counts"].tolist(), [0])
        np.testing.assert_array_equal(empty["recent_ring"], -np.ones((1,1,2), np.int64))
        hot = partition_reference([[0, 1, 2, -1]], positions=[2], tags=tags_through(3, 256))
        self.assertEqual(hot["recent_counts"].tolist(), [3])
        self.assertFalse(hot["old_present"][0])
        disabled = partition_reference([[0, 1, 2, 9, -1]], positions=[2], window=0,
                                       tags=object(), capacity=object())
        self.assertEqual(disabled["old"][0,0,:3].tolist(), [0, 1, 2])
        self.assertEqual(disabled["recent_ring"].shape, (1,1,0))
        self.assertFalse(disabled["recent_present"][0])

    def test_invalid_topk_format_and_duplicate_mass_are_rejected(self):
        for row in ((0,-1,1), (0,-2,-1), (1,1,-1)):
            with self.subTest(row=row), self.assertRaises(ContractError):
                partition_reference([row], positions=[2], tags=tags_through(3, 256))
        with self.assertRaises(ContractError):
            partition_reference([[1.]], positions=[2], tags=tags_through(3, 256))
        # IDs above FP32's exact integer range remain exact.
        p = 2**24+300
        out = partition_reference([[p-200, p]], positions=[p], tags=tags_through(p+1, 256))
        self.assertEqual(int(out["old"][0,0,0]), p-200)
        self.assertEqual(int(out["recent_global"][0,0,0]), p)

    def test_fused_modes_report_every_fallback_and_require_real_library(self):
        config = parse_recent128_config({"sfa_recent128": dict(window=128, reader="fused_required",
            library_dir="/tmp/current-direct-library", source_root="/tmp/operator-source",
            diagnostics_path="/tmp/recent128.json")})
        args = dict(hot_count=128, heads=64, latent_dim=512, rope_dim=64,
                    query_count=32, library_available=True)
        self.assertTrue(fusion_eligibility(config, **args).use_fused)
        for key, value in (("hot_count", 127), ("heads", 32), ("latent_dim", 1024),
                           ("rope_dim", 0), ("query_count", 1025)):
            decision = fusion_eligibility(config, **dict(args, **{key:value}))
            self.assertFalse(decision.use_fused)
            self.assertIn(key, decision.reason)
        with self.assertRaisesRegex(ContractError, "real direct library"):
            fusion_eligibility(config, **dict(args, library_available=False))
        eligible = Recent128Config(window=128, reader="fused_eligible", library_dir="/tmp/lib",
                                   source_root="/tmp/src", diagnostics_path="/tmp/recent128.json")
        self.assertEqual(fusion_eligibility(eligible, **dict(args, library_available=False)).reason,
                         "library_unavailable")

    def test_single_pass_config_requires_paths_and_bypasses_old_hot128_abi4_gate(self):
        values = dict(window=128, reader="single_pass", library_dir="/tmp/single-pass-build",
                      source_root="/tmp/operator-source", diagnostics_path="/tmp/recent128-single-pass",
                      preprocess="native_accuracy")
        cfg = parse_recent128_config({"sfa_recent128": values})
        self.assertEqual(cfg.reader, "single_pass")
        self.assertEqual(cfg.single_pass_kernel,"overlay")
        fast = parse_recent128_config({"sfa_recent128":dict(values,single_pass_kernel="fast_pair")})
        self.assertEqual(fast.single_pass_kernel,"fast_pair")
        load = parse_recent128_config({"sfa_recent128":dict(values,single_pass_kernel="load_tile4")})
        self.assertEqual(load.single_pass_kernel,"load_tile4")
        with self.assertRaisesRegex(ContractError,"single_pass_kernel"):
            parse_recent128_config({"sfa_recent128":dict(values,single_pass_kernel="early_hot")})
        self.assertEqual(cfg.preprocess, "native_accuracy")
        for name in ("library_dir", "source_root", "diagnostics_path"):
            incomplete = dict(values)
            del incomplete[name]
            with self.subTest(missing=name), self.assertRaises(ContractError):
                parse_recent128_config({"sfa_recent128": incomplete})
        # This helper delegates to the dedicated whole-TopK runtime; it cannot
        # bless an old ABI4 hot merge even when that old count128 gate matches.
        for count in (0, 1, 127, 128):
            decision = fusion_eligibility(cfg, hot_count=count, heads=64, latent_dim=512,
                                          rope_dim=64, query_count=4, library_available=True)
            self.assertFalse(decision.use_fused)
            self.assertEqual(decision.reason, "single_pass_whole_topk_runtime")
        delegated = fusion_eligibility(cfg, hot_count=128, heads=32, latent_dim=512,
                                      rope_dim=0, query_count=4, library_available=False)
        self.assertEqual(delegated.reason, "single_pass_whole_topk_runtime")


if __name__ == "__main__":
    unittest.main()

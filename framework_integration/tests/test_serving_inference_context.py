"""Regression for clearing an inference-owned graph plan from normal mode.

Uses a strict inference-tensor fake so the original unguarded method actually
fails. No Torch import, NPU allocation, or device execution is needed.
"""
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch


class ServingInferenceContextTest(unittest.TestCase):
    def setUp(self):
        self.mode = False
        self.enters = 0
        testcase = self
        torch = ModuleType("torch")
        torch.is_inference_mode_enabled = lambda: testcase.mode
        @contextmanager
        def inference_mode():
            previous = testcase.mode
            testcase.enters += 1
            testcase.mode = True
            try:
                yield
            finally:
                testcase.mode = previous
        torch.inference_mode = inference_mode
        self.guard = patch.dict(sys.modules, {"torch": torch})
        self.guard.start()
        self.addCleanup(self.guard.stop)
        source = Path(__file__).resolve().parents[1] / "sfa_recent128/serving_reader.py"
        spec = importlib.util.spec_from_file_location("serving_inference_regression", source)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_reset_after_graph_preserves_storage_and_is_independent_of_caller_mode(self):
        testcase = self
        class Plan:
            def __init__(self):
                self.fault_storage = object()
                self.reset_calls = 0
            def reset_faults(self):
                if not testcase.mode:
                    raise RuntimeError("Inplace update to inference tensor outside InferenceMode")
                self.reset_calls += 1
        plan = Plan()
        reader = object.__new__(self.module.ServingReader)
        reader.prepared = {"warmup": {"plans": (plan,)}, "capture_alias": {"plans": (plan,)}}
        original_storage = plan.fault_storage
        with self.assertRaisesRegex(RuntimeError, "outside InferenceMode"):
            self.module.ServingReader.reset_faults.__wrapped__(reader)
        reader.reset_faults()
        self.assertEqual(plan.reset_calls, 1)
        self.assertIs(plan.fault_storage, original_storage)
        self.assertFalse(self.mode)
        self.assertEqual(self.enters, 1)
        self.assertIs(reader.prepared["warmup"]["plans"][0], plan)

    def test_already_enabled_path_does_not_reenter_and_exception_restores_mode(self):
        @self.module._inference_context
        def mutate(fail=False):
            self.assertTrue(self.mode)
            if fail:
                raise RuntimeError("expected control")
        with self.assertRaisesRegex(RuntimeError, "expected control"):
            mutate(True)
        self.assertFalse(self.mode)
        self.assertEqual(self.enters, 1)
        self.mode = True
        mutate()
        self.assertEqual(self.enters, 1)
        self.assertTrue(self.mode)
        self.assertTrue(hasattr(self.module.ServingReader.read, "__wrapped__"))


if __name__ == "__main__":
    unittest.main()

"""Focused runner/builder control-flow tests with host-only tensor fakes."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfa_recent128 import serving_runner
from sfa_recent128.serving_state import ServingState
from test_serving_state_lifecycle import HostTorch, Tensor


class ServingRunnerTests(unittest.TestCase):
    def setUp(self):
        self.t = HostTorch()
        self.state = ServingState(request_pool_size=8, request_table_capacity=32,
            max_tokens=32, max_query_per_request=4, device="npu:0", torch_module=self.t)
        self.state.slot_valid = self.t.empty((32,), dtype=self.t.bool, device="npu:0")
        self.state.register_layer("layer")
        # Root's hook allocates each lane's valid buffer during metadata warmup.
        def new_empty(tensor, shape):
            return self.t.empty(shape, dtype=tensor.dtype, device=tensor.device)
        self.new_empty = patch.object(Tensor, "new_empty", new=new_empty, create=True)
        self.new_empty.start()
        self.addCleanup(self.new_empty.stop)
        self.vc = SimpleNamespace(additional_config={"sfa_recent128": dict(
            window=128, execution="serving", preprocess="preserve", reader="single_pass",
            single_pass_kernel="batched_v4", max_query_chunk=4,
            library_dir="/tmp/fake-lib", source_root="/tmp/fake-src", diagnostics_path="/tmp/fake-report")},
            model_config=SimpleNamespace(model="host-fixture"),
            scheduler_config=SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=32))
        self.events = []
        events, vc = self.events, self.vc

        class Builder:
            def __init__(builder):
                builder.vllm_config, builder.device = vc, "npu:0"

            def _build(builder, common_attn_metadata, draft_index=None):
                events.append(("builder_original", draft_index))
                # Older reviewed public VA does not retain positions/boundaries.
                return SimpleNamespace(num_input_tokens=common_attn_metadata.num_input_tokens,
                    num_actual_tokens=common_attn_metadata.num_actual_tokens,
                    slot_mapping=common_attn_metadata.slot_mapping)

        class Runner:
            def __init__(runner):
                runner.vllm_config, runner.device = vc, "npu:0"
                runner.input_batch = SimpleNamespace(req_ids=["request-a", "request-b"],
                    num_computed_tokens_cpu=[0, 0])
                runner.builder = Builder()
                runner.common = None

            def _build_attention_metadata(runner, num_tokens, num_reqs, max_query_len,
                                          for_cudagraph_capture=False):
                events.append(("runner_original", num_tokens, num_reqs))
                return runner.builder._build(runner.common), "original-spec-result"

            def execute_model(runner, scheduler_output, *args, **kwargs):
                events.append(("execute_original", scheduler_output))
                return "original-execute-result"

        self.runner_class, self.builder_class = Runner, Builder
        serving_runner._LAST.clear()
        serving_runner.install_runner_hooks(Runner, Builder)
        self.runner = Runner()
        self.state_patch = patch.object(serving_runner, "get_state", return_value=self.state)
        self.state_patch.start()
        self.addCleanup(self.state_patch.stop)

    def common(self, positions, boundaries, *, actual=None):
        return SimpleNamespace(positions=self.t.tensor(positions),
            query_start_loc=self.t.tensor(boundaries),
            slot_mapping=self.t.tensor([0] * len(positions)),
            num_input_tokens=len(positions),
            num_actual_tokens=len(positions) if actual is None else actual,
            max_query_len=4)

    def test_stable_scheduler_identity_corrected_positions_and_original_results(self):
        self.runner.common = self.common([0, 1, 0, 0], [0, 2, 3], actual=3)
        metadata, spec = self.runner._build_attention_metadata(3, 2, 4)
        self.assertEqual(spec, "original-spec-result")
        first_lane = metadata.recent128_state
        self.assertIs(first_lane.pool_owner, self.state)
        self.assertEqual(metadata.recent128_token_banks.data.tolist(), [0, 0, 1, -1])
        self.assertEqual(metadata.recent128_token_valid.data.tolist(), [True, True, True, False])
        self.runner.input_batch.req_ids = ["request-b", "request-a"]
        # Optimistic host starts deliberately differ from corrected device data.
        self.runner.input_batch.num_computed_tokens_cpu = [90, 100]
        self.runner.common = self.common([4, 3], [0, 1, 2])
        changed, _ = self.runner._build_attention_metadata(2, 2, 4)
        self.assertIs(changed.recent128_state, first_lane)
        self.assertEqual(changed.recent128_token_banks.data.tolist(), [1, 0])
        self.assertEqual(changed.recent128_token_positions.data.tolist(), [4, 3])
        # Draft builder calls outside runner scope use the latest scheduler IDs,
        # with their own corrected positions and independent route memory.
        draft = self.runner.builder._build(self.common([5, 4], [0, 1, 2]), draft_index=0)
        self.assertIsNot(draft.recent128_state, first_lane)
        self.assertEqual(draft.recent128_token_positions.data.tolist(), [5, 4])
        self.assertEqual(changed.recent128_token_positions.data.tolist(), [4, 3])
        self.assertTrue(any(event[0] == "runner_original" for event in self.events))
        self.assertTrue(any(event[0] == "builder_original" for event in self.events))

    def test_capture_dummy_creates_no_real_request_and_replay_bucket_reuses_routes(self):
        self.runner.common = self.common([0, 0, 0, 0], [0, 2, 4])
        dummy, _ = self.runner._build_attention_metadata(4, 2, 4, for_cudagraph_capture=True)
        lane = dummy.recent128_state
        self.assertFalse(dummy.recent128_token_valid.data.any())
        self.assertEqual(self.state.requests, {})
        self.runner.common = self.common([0, 1, 0, 0], [0, 2, 3], actual=3)
        real, _ = self.runner._build_attention_metadata(3, 2, 4)
        self.assertIs(real.recent128_state, lane)
        self.assertEqual(real.recent128_token_valid.data.tolist(), [True, True, True, False])

    def test_off_delegates_and_finished_request_release_runs_before_original_execute(self):
        self.runner.common = self.common([0, 0], [0, 1, 2])
        self.runner._build_attention_metadata(2, 2, 4)
        result = self.runner.execute_model(SimpleNamespace(finished_req_ids={"request-a"}))
        self.assertEqual(result, "original-execute-result")
        self.assertNotIn("request-a", self.state.requests)
        self.vc.additional_config = {}
        off, original_spec = self.runner._build_attention_metadata(2, 2, 4)
        self.assertFalse(hasattr(off, "recent128_state"))
        self.assertEqual(original_spec, "original-spec-result")


if __name__ == "__main__":
    unittest.main()

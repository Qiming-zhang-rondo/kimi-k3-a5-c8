"""Bind scheduler identity to corrected device metadata once per SFA build.

Hooks target the reviewed public VA runner/builder interfaces, not tensor
contents. Drafting uses the same persistent scheduler identities and the
builder's NEW device positions; host optimistic lengths never replace them.
Prefix/PD integrations call receive_tail before admitting a nonzero start.
"""
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import importlib

from .contract import parse_recent128_config
from .serving_runtime import get_state, state_key

_ACTIVE = ContextVar("recent128_runner_build", default=None)
_DUMMY = ContextVar("recent128_runner_dummy", default=False)
_LAST = {}


@dataclass
class RunnerStep:
    state: object
    request_ids: tuple
    starts: dict
    dummy: bool


def _selected(vc):
    config = parse_recent128_config(getattr(vc, "additional_config", None))
    return config if config.enabled and config.execution == "serving" else None


def receive_tail(vllm_config, device, request_id, snapshot, *, source, completion=None):
    """Connector/prefix callback after actual BF16 tail delivery, before build.

    Does not mark C4 arrival as BF16 arrival. Never manufactures missing data
    by dequantizing the cold cache. The completion orders the receiving stream.
    """
    config = _selected(vllm_config)
    if config is None:
        return None
    state = get_state(vllm_config, config, device)
    state.acquire_request(request_id, initial_position=snapshot.end)
    state.import_tail(request_id, snapshot, source=source, completion=completion)
    return state


def snapshot_tail(vllm_config, device, request_id, *, committed_end, completion=None):
    """Connector/prefix callback at an actual committed boundary, outside Graph."""
    config = _selected(vllm_config)
    if config is None:
        return None
    state = get_state(vllm_config, config, device)
    state.commit_request(request_id, end=committed_end)
    return state.export_tail(request_id, end=committed_end, completion=completion)


def bind_metadata(metadata, context):
    state = context.state
    positions = getattr(metadata, "positions", None)
    boundaries = getattr(metadata, "query_start_loc", None)
    slots = getattr(metadata, "slot_mapping", None)
    if positions is None or boundaries is None or slots is None:
        raise RuntimeError("SFA metadata lacks positions/query_start_loc/slot_mapping required by recent128 serving")
    count = int(positions.shape[0])
    if int(getattr(metadata, "max_query_len", 0)) > state.max_query_per_request:
        raise RuntimeError("Set recent128.max_query_chunk to the real scheduler per-request chunk bound before allocating rings")
    if count > state.max_tokens:
        raise RuntimeError("Recent128 token bucket exceeds the runner's preallocated token capacity")
    ids = () if context.dummy else context.request_ids
    if boundaries.shape[0] < len(ids) + 1:
        raise RuntimeError("Corrected SFA query boundaries do not describe the current scheduler requests")
    valid = state.slot_valid[:count]
    state.torch.ge(slots[:count], 0, out=valid)
    # Profiling/capture dummy tokens must not create real request ownership.
    if context.dummy:
        valid.fill_(False)
    state.bind_step(ids, positions=positions,
        query_start_loc=boundaries[:len(ids) + 1], valid=valid,
        num_actual_tokens=0 if context.dummy else int(metadata.num_actual_tokens),
        request_starts=context.starts, token_count=count)
    state.attach_metadata(metadata)
    return metadata


def install_runner_hooks(runner_class=None, builder_class=None):
    if runner_class is None:
        runner_class = importlib.import_module("vllm_ascend.worker.model_runner_v1").NPUModelRunner
    if builder_class is None:
        builder_class = importlib.import_module("vllm_ascend.attention.sfa_v1").AscendSFAMetadataBuilder
    if not getattr(runner_class, "_recent128_serving_runner", False):
        original = getattr(runner_class, "_build_attention_metadata", None)
        execute = getattr(runner_class, "execute_model", None)
        if not callable(original) or not callable(execute):
            raise RuntimeError("Reviewed NPUModelRunner metadata/execute interfaces are unavailable")

        @wraps(original)
        def build(runner, *args, **kwargs):
            config = _selected(runner.vllm_config)
            if config is None:
                return original(runner, *args, **kwargs)
            state = get_state(runner.vllm_config, config, runner.device)
            requested = kwargs.get("num_reqs", args[1] if len(args) > 1 else None)
            real_ids = tuple(runner.input_batch.req_ids)
            if requested is not None:
                real_ids = real_ids[:int(requested)]
            dummy = _DUMMY.get() or bool(kwargs.get("for_cudagraph_capture", False)) or not real_ids
            starts = {request_id: int(runner.input_batch.num_computed_tokens_cpu[row])
                      for row, request_id in enumerate(real_ids)}
            context = RunnerStep(state, real_ids, starts, dummy)
            key = state_key(runner.vllm_config, config, runner.device)
            _LAST[key] = context
            token = _ACTIVE.set(context)
            try:
                return original(runner, *args, **kwargs)
            finally:
                _ACTIVE.reset(token)

        @wraps(execute)
        def execute_model(runner, scheduler_output, *args, **kwargs):
            config = _selected(runner.vllm_config)
            if config is not None:
                state = get_state(runner.vllm_config, config, runner.device)
                if getattr(runner.vllm_config, "kv_transfer_config", None) is not None:
                    from vllm.distributed.kv_transfer import get_kv_transfer_group
                    from .pd_mooncake import pin_finished_tails
                    pin_finished_tails(get_kv_transfer_group(),
                        getattr(scheduler_output, "kv_connector_metadata", None))
                for request_id in getattr(scheduler_output, "finished_req_ids", ()):
                    state.release_request(request_id)
            return execute(runner, scheduler_output, *args, **kwargs)

        runner_class._build_attention_metadata = build
        runner_class.execute_model = execute_model
        original_dummy = getattr(runner_class, "_dummy_run", None)
        if callable(original_dummy):
            @wraps(original_dummy)
            def dummy_run(runner, *args, **kwargs):
                token = _DUMMY.set(True)
                try:
                    return original_dummy(runner, *args, **kwargs)
                finally:
                    _DUMMY.reset(token)
            runner_class._dummy_run = dummy_run
        runner_class._recent128_serving_runner = True
    if not getattr(builder_class, "_recent128_serving_builder", False):
        original_build = getattr(builder_class, "_build", None)
        if not callable(original_build):
            raise RuntimeError("Reviewed AscendSFAMetadataBuilder._build interface is unavailable")

        @wraps(original_build)
        def metadata_build(builder, *args, **kwargs):
            metadata = original_build(builder, *args, **kwargs)
            config = _selected(builder.vllm_config)
            if config is None:
                return metadata
            context = _ACTIVE.get()
            if context is None:
                context = _LAST.get(state_key(builder.vllm_config, config, builder.device))
            if context is None:
                raise RuntimeError("Recent128 drafting metadata lacks scheduler request identity; runner hook did not execute")
            common = kwargs.get("common_attn_metadata", args[0] if args else None)
            # Some reviewed VA revisions do not retain these corrected fields
            # on SFA metadata. Copy REFERENCES/views from this exact build,
            # never reconstruct positions from host optimistic lengths.
            count = int(metadata.num_input_tokens)
            if getattr(metadata, "positions", None) is None:
                metadata.positions = common.positions[:count]
            if getattr(metadata, "query_start_loc", None) is None:
                metadata.query_start_loc = common.query_start_loc
            if not getattr(metadata, "max_query_len", 0):
                metadata.max_query_len = int(common.max_query_len)
            draft_index = kwargs.get("draft_index", args[1] if len(args) > 1 else None)
            lane = context.state.routing_lane((id(builder), draft_index))
            if not hasattr(lane, "slot_valid"):
                lane.slot_valid = context.state.slot_valid.new_empty((lane.max_tokens,))
            return bind_metadata(metadata, RunnerStep(lane, context.request_ids, context.starts, context.dummy))

        builder_class._build = metadata_build
        builder_class._recent128_serving_builder = True
    return dict(status="INSTALLED", runner=runner_class.__name__, builder=builder_class.__name__)

"""Native BF16 producer plans for persistent serving lanes.

Importing and constructing this controller does not import a device framework.
The first eager invocation prepares exchange buffers, local query routes and a
writer using the actual captured source. Graph capture only reuses those plans.
"""
from dataclasses import dataclass
from types import SimpleNamespace

from .native_cp import (DSACPLayout, NativeCPBuffers, NativeCPError,
                        NativeQueryRoutes, gather_native_rows)
from .native_writer import PreparedNativeWriter


_RING_FIELDS = ("recent_latent", "recent_rope", "recent_position_tags",
                "recent_epoch_tags", "token_ring_slots", "token_positions",
                "token_request_epochs", "token_valid")


def _require(condition, message):
    if not condition:
        raise NativeCPError(message)


def _is_capturing():
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX
    return bool(_EXTRA_CTX.capturing)


def _tensor_signature(tensor):
    """Only static metadata; never inspect routing or payload values."""
    stride = getattr(tensor, "stride", None)
    return (tuple(getattr(tensor, "shape", ())),
            tuple(stride()) if callable(stride) else None,
            str(getattr(tensor, "dtype", None)),
            str(getattr(tensor, "device", None)))


def _binding_signature(binding):
    return (tuple((id(getattr(binding, name)),
                   _tensor_signature(getattr(binding, name)))
                  for name in _RING_FIELDS), binding.max_query_per_request)


@dataclass
class _NativePlan:
    lane: object
    layout: DSACPLayout
    binding: object
    buffers: object
    routes: object
    group: object
    writer: object = None
    written: bool = False


class NativeProducer:
    """One layer's full-ring producer and local-query routing controller.

    The TP token gather is enabled only by an actual DSACPContext. Ordinary
    native attention already owns full replicated KV rows, even with TP>1.
    Plans hold their lane and source binding alive, preventing identity reuse.
    """

    def __init__(self, runtime):
        self.runtime = runtime
        self._plans = {}
        self._written_plans = {}
        self._torch = None
        self._tp_group = None

    def _bound(self, metadata):
        lane = getattr(metadata, "recent128_state", None)
        _require(lane is not None and getattr(lane, "pool_owner", lane) is self.runtime.state,
                 "recent128 native producer lacks this step's bound request metadata")
        _require(getattr(metadata, "max_query_len", 0) <= self.runtime.config.max_query_chunk,
                 "Recent128 capacity bound is smaller than the scheduler query chunk")
        count = getattr(metadata, "num_input_tokens", None)
        _require(type(count) is int and count > 0,
                 "Native metadata.num_input_tokens must be a positive global token count")
        return lane, lane.binding(self.runtime.layer_key, token_rows=count), count

    def _layout(self, metadata, count, *, capturing, prepare=False):
        context = getattr(metadata, "dsa_cp_context", None)
        if context is None:
            context = SimpleNamespace(num_tokens=count, num_tokens_pad=count,
                                      local_start=0, local_end=count,
                                      local_end_with_pad=count)
            return DSACPLayout.from_context(context, global_tokens=count,
                                            tp_size=1, tp_rank=0), None
        if self._tp_group is None:
            _require(prepare,
                     "Native reader requires a prepared global producer write for this lane and bucket")
            _require(not capturing,
                     "Warm up this native DSACP lane before Graph capture; TP group is unprepared")
            # Resolve the same group used by the original packed-KV gather.
            from vllm_ascend.attention.context_parallel.sfa_cp import get_tp_group
            self._tp_group = get_tp_group()
        group = self._tp_group
        return DSACPLayout.from_context(context, global_tokens=count,
                                        tp_size=group.world_size,
                                        tp_rank=group.rank_in_group), group.device_group

    def write(self, scope):
        """Gather the real tap once and publish one global ring write."""
        _require(scope.impl is self.runtime.impl, "Native capture belongs to another layer implementation")
        _require(getattr(scope, "writes", 0) == 0,
                 "Native forward requires exactly one recent128 producer write")
        latent, rope = scope.require_capture()
        capturing = _is_capturing()
        lane, binding, count = self._bound(scope.metadata)
        layout, group = self._layout(scope.metadata, count, capturing=capturing, prepare=True)
        route_key = (id(lane), layout, _binding_signature(binding))
        key = (route_key, _tensor_signature(latent), _tensor_signature(rope))
        plan = self._plans.get(key)
        if plan is None:
            _require(not capturing,
                     "Warm up this native lane and batch bucket before Graph capture; no plan allocation inside capture")
            if self._torch is None:
                import torch
                self._torch = torch
            buffers = NativeCPBuffers.prepare(layout, device=latent.device,
                                               torch_module=self._torch)
            routes = NativeQueryRoutes.prepare(layout, binding, torch_module=self._torch)
            plan = _NativePlan(lane, layout, binding, buffers, routes, group)
            self._plans[key] = plan
        _require(not capturing or (plan.writer is not None and plan.written),
                 "Native writer must be prepared and warmed before Graph capture")
        gathered_latent, gathered_rope = gather_native_rows(
            latent, rope, layout, plan.buffers, torch_module=self._torch,
            group=plan.group)
        if plan.writer is None:
            _require(not capturing, "Native writer must be prepared and warmed before Graph capture")
            # Preparation validates the gathered source's real strides. No
            # fabricated placeholder values or C4 cache enter the writer.
            plan.writer = PreparedNativeWriter(gathered_latent, gathered_rope,
                                               plan.binding, check_is_capture=_is_capturing)
        result = plan.writer.write(gathered_latent, gathered_rope)
        plan.written = True
        self._written_plans[route_key] = plan
        scope.writes = getattr(scope, "writes", 0) + 1
        self.runtime.producer_calls += 1
        return result

    def reader_binding(self, metadata, token_rows):
        """Return routes for local queries after their global ring write."""
        lane, binding, count = self._bound(metadata)
        layout, _ = self._layout(metadata, count, capturing=_is_capturing())
        _require(type(token_rows) is int and token_rows == layout.local_rows,
                 "Native reader query rows must match the local DSACP layout")
        key = (id(lane), layout, _binding_signature(binding))
        plan = self._written_plans.get(key)
        _require(plan is not None and plan.written,
                 "Native reader requires a prepared global producer write for this lane and bucket")
        plan.routes.update()
        return plan.routes.binding

    def configuration(self):
        return dict(producer="native_bf16_candidate", plans=len(self._plans),
            cp_tp_sizes=sorted({plan.layout.tp_size for plan in self._plans.values()}),
            extra_bf16_collectives_per_layer_call=1 if any(plan.layout.tp_size>1 for plan in self._plans.values()) else 0,
            device_validation="PENDING", performance_claim=False)

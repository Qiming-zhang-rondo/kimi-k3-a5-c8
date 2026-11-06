# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental MXFP8 QFA over temporarily expanded K3 Prefill KV.

CANN 9.2 supports QFA D=256 and requires equal Q/K/V dimensions. Pad the
expanded QK192/V128 with zeros, retain the model's original softmax scale,
and discard the extra output columns. The persistent latent576 FP8 cache
and QFM Decode contract are unchanged. This introduces another quantization
of temporary expanded tensors and requires separate model accuracy testing.
"""

import importlib
import inspect
from contextlib import suppress
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
import torch_npu

QFA_DIM = 256
QFA_QUERY_DIM = 192
QFA_VALUE_DIM = 128
QFA_QUANT_MODE = 1
QFA_GROUP = 32
QFA_SCALE_PAIR = 64


@dataclass
class PreparedQFAQuery:
    values: torch.Tensor
    descales: torch.Tensor
    cu_query: torch.Tensor
    # Step-local plans may be reused for equally sized history chunks. They
    # never survive a prepared query or retain per-layer KV tensors.
    plans: dict = field(default_factory=dict)


def _validate_abi(operator, positional_names, keywords, optional_names, name):
    if not callable(operator):
        raise RuntimeError(f"Expanded QFA requires callable CANN 9.2 {name}")
    source = None
    signature = None
    with suppress(TypeError):
        source = inspect.getsourcefile(operator)
    try:
        signature = inspect.signature(operator)
        # The delivered 9.2 Python wrappers have explicit named arguments.
        # An unknown variadic shim must not silently bypass this ABI check.
        if any(
            value.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            for value in signature.parameters.values()
        ):
            raise ValueError("variadic wrapper does not expose the required ABI")
        expected = set((*positional_names, *keywords, *optional_names))
        found = set(signature.parameters)
        if found != expected:
            raise ValueError(
                f"unexpected parameters={sorted(found - expected)}, missing parameters={sorted(expected - found)}"
            )
        signature.bind(*([None] * len(positional_names)), **dict.fromkeys(keywords))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Expanded QFA requires the fixed CANN 9.2 {name} ABI: {exc}; "
            f"loaded source={source}, signature={signature}. No newer tail ABI or FIA fallback is selected."
        ) from exc
    return {"source": source, "signature": str(signature)}


class ExpandedQFA:
    """Use the explicit CANN QFA/metadata wrappers, with no FIA fallback."""

    def __init__(self):
        try:
            ops = importlib.import_module("cann_ops_transformer.ops")
            self.attention = ops.quant_flash_attn
            self.metadata = ops.quant_flash_attn_metadata
        except (ImportError, AttributeError) as exc:
            raise RuntimeError("Expanded QFA needs installed CANN 9.2 quant_flash_attn and metadata wrappers") from exc
        common = (
            "cu_seqlens_q",
            "cu_seqlens_kv",
            "seqused_q",
            "seqused_kv",
            "max_seqlen_q",
            "max_seqlen_kv",
            "mask_mode",
            "win_left",
            "win_right",
            "layout_q",
            "layout_q_descale",
            "layout_kv",
            "layout_out",
        )
        self.abi = {
            "metadata": _validate_abi(
                self.metadata,
                ("num_heads_q", "num_heads_kv", "head_dim", "quant_mode"),
                common,
                ("batch_size", "head_dim_v", "is_grad_enabled"),
                "quant_flash_attn_metadata",
            ),
            "attention": _validate_abi(
                self.attention,
                ("q", "k", "v", "q_descale", "k_descale", "v_descale", "quant_mode"),
                (
                    *common,
                    "block_table",
                    "p_scale",
                    "sinks",
                    "attn_mask",
                    "metadata",
                    "softmax_scale",
                    "return_softmax_lse",
                ),
                (),
                "quant_flash_attn",
            ),
        }
        if not callable(getattr(torch_npu, "npu_dynamic_mx_quant", None)):
            raise RuntimeError("Expanded QFA requires torch_npu.npu_dynamic_mx_quant")

    @staticmethod
    def _quantize_last_dim(value):
        values, scales = torch_npu.npu_dynamic_mx_quant(value, dst_type=torch.float8_e4m3fn, scale_alg=0)
        expected_shape = (*value.shape[:-1], value.shape[-1] // QFA_SCALE_PAIR, 2)
        if (
            values.dtype != torch.float8_e4m3fn
            or values.shape != value.shape
            or scales.element_size() != 1
            or scales.numel() != value.numel() // QFA_GROUP
        ):
            raise RuntimeError("Expanded QFA dynamic MXFP8 returned an incompatible value/scale layout")
        # Scale storage is E8M0 bytes. Byte copies avoid float8 layout
        # conversions that can fall back to AICPU on some PTA deliveries.
        values = values.view(torch.uint8).contiguous().view(torch.float8_e4m3fn)
        scales = scales.view(torch.uint8).reshape(expected_shape).contiguous().view(torch.float8_e8m0fnu)
        return values, scales

    def prepare_query(self, query):
        """Pad and quantize Q once per request, reusing it for every KV chunk."""
        if (
            query.ndim != 3
            or query.dtype != torch.bfloat16
            or query.shape[-1] != QFA_QUERY_DIM
            or query.shape[0] <= 0
            or not 0 < query.shape[1] <= 256
        ):
            raise ValueError("Expanded QFA Query must be nonempty BF16 [T,heads<=256,192]")
        values, scales = self._quantize_last_dim(F.pad(query, (0, QFA_DIM - QFA_QUERY_DIM)).contiguous())
        cu = torch.tensor([0, query.shape[0]], dtype=torch.int32, device=query.device)
        return PreparedQFAQuery(values, scales, cu)

    def quantize_kv(self, key, value):
        """Return actual encoded K/V and scales, useful for independent probes.

        Q/K share scales along their D axis. V shares scales along its token
        axis, with two Group32 scales per 64-token group. Every call describes
        one request, so V groups cannot straddle request boundaries.
        """
        if (
            key.ndim != 3
            or value.ndim != 3
            or key.shape[:2] != value.shape[:2]
            or key.shape[-1] != QFA_QUERY_DIM
            or value.shape[-1] != QFA_VALUE_DIM
            or key.shape[0] <= 0
            or key.dtype != torch.bfloat16
            or value.dtype != torch.bfloat16
            or key.device != value.device
        ):
            raise ValueError("Expanded QFA KV must be nonempty BF16 K[T,H,192] and V[T,H,128] on one device")
        key8, key_scale = self._quantize_last_dim(F.pad(key, (0, QFA_DIM - QFA_QUERY_DIM)).contiguous())
        tokens, heads = key.shape[:2]
        padded_tokens = (tokens + QFA_SCALE_PAIR - 1) // QFA_SCALE_PAIR * QFA_SCALE_PAIR
        # Do not retain an additional BF16 TND padded V while quantizing its
        # token-last copy. Neither temporary survives this quantization call.
        value8, value_scale = self._quantize_last_dim(
            F.pad(value, (0, QFA_DIM - QFA_VALUE_DIM, 0, 0, 0, padded_tokens - tokens)).permute(1, 2, 0).contiguous()
        )
        value8 = value8.view(torch.uint8).permute(2, 0, 1).contiguous()[:tokens].view(torch.float8_e4m3fn)
        value_scale = value_scale.view(torch.uint8).permute(2, 0, 1, 3).contiguous().view(torch.float8_e8m0fnu)
        if value_scale.shape != (padded_tokens // QFA_SCALE_PAIR, heads, QFA_DIM, 2):
            raise RuntimeError("Expanded QFA V scales do not match token-axis Group32 layout")
        return key8, value8, key_scale, value_scale

    def partial(self, prepared, key, value, scale, mask=None, *, metadata_cache=None):
        """Compute one history/current contribution and natural-log LSE."""
        query = prepared.values
        if key.ndim != 3 or key.shape[1] != query.shape[1] or key.device != query.device:
            raise ValueError("Expanded QFA KV heads/device must match the DCP-gathered Query")
        if mask is not None and (
            mask.shape != (2048, 2048)
            or mask.device != query.device
            or mask.dtype not in (torch.int8, torch.uint8, torch.bool)
        ):
            raise ValueError("Expanded QFA causal mask must be INT8/UINT8/BOOL [2048,2048] on the Query device")
        key8, value8, key_scale, value_scale = self.quantize_kv(key, value)
        query_tokens, heads = query.shape[:2]
        kv_tokens = key.shape[0]
        mode = 3 if mask is not None else 0
        plan_key = "expanded_qfa", query.device, heads, query_tokens, kv_tokens, QFA_DIM, mode
        plans = prepared.plans if metadata_cache is None else metadata_cache
        plan = plans.get(plan_key)
        if plan is None:
            cu_kv = torch.tensor([0, kv_tokens], dtype=torch.int32, device=query.device)
            attrs = dict(
                cu_seqlens_q=prepared.cu_query,
                cu_seqlens_kv=cu_kv,
                seqused_q=None,
                seqused_kv=None,
                max_seqlen_q=query_tokens,
                max_seqlen_kv=kv_tokens,
                mask_mode=mode,
                win_left=-1,
                win_right=-1,
                layout_q="TND",
                layout_q_descale="TND",
                layout_kv="TND",
                layout_out="TND",
            )
            schedule = self.metadata(heads, heads, QFA_DIM, QFA_QUANT_MODE, **attrs)
            plan = schedule, attrs
            plans[plan_key] = plan
        schedule, attrs = plan
        result = self.attention(
            query,
            key8,
            value8,
            prepared.descales,
            key_scale,
            value_scale,
            QFA_QUANT_MODE,
            block_table=None,
            p_scale=None,
            sinks=None,
            attn_mask=None if mask is None else mask.to(torch.int8),
            metadata=schedule,
            softmax_scale=scale,
            return_softmax_lse=True,
            **attrs,
        )
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise RuntimeError("Expanded QFA must return output and natural-log LSE; no fallback is permitted")
        output, lse = result
        if (
            output.shape != (query_tokens, heads, QFA_DIM)
            or lse.shape != (heads, query_tokens)
            or output.dtype != torch.bfloat16
            or lse.dtype != torch.float32
        ):
            raise RuntimeError("Expanded QFA returned an incompatible TND output/LSE layout")
        return output[:, :, :QFA_VALUE_DIM].float(), lse.transpose(0, 1).unsqueeze(-1).float()

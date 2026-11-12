# SPDX-License-Identifier: Apache-2.0
"""Private ND projection weights for K3's full576 QFM DSL prolog.

ModelSlim has already transposed weights and paired the MX scales before this
adapter runs. Never mutate those tensors: the ordinary path still uses them.
"""

import math

import torch
import torch_npu

MIN_HIDDEN_SIZE = 1024
MAX_HIDDEN_SIZE = 8192
HIDDEN_ALIGNMENT = 256
Q_RANKS = (640, 1024, 1536, 2048)
KV_RANK = 512
Q_NOPE_DIMS = (128, 192)
RAW_DIM = 64
MX_GROUP_SIZE = 32
MX_SCALE_PAIR = 2
ACL_FORMAT_ND = 2


def _projection(layer, name, inputs, outputs):
    # Lazy import avoids initializing the quantization/attention import cycle
    # when this independently testable adapter is imported.
    from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import AscendW8A8MXFP8DynamicLinearMethod

    method = getattr(layer, "quant_method", None)
    method = getattr(method, "quant_method", method)
    if not isinstance(method, AscendW8A8MXFP8DynamicLinearMethod) or method.group_size != MX_GROUP_SIZE:
        raise ValueError(f"QFM prolog {name} requires ModelSlim W8A8_MXFP8 Group32")
    if getattr(method, "dynamic_mx_quant_scale_alg", None) != 0:
        raise ValueError(f"QFM prolog {name} requires dynamic_mx_quant_scale_alg=0 to match the native QR quantizer")
    if not getattr(layer, "_mxfp8_transformed", False):
        raise ValueError(f"QFM prolog {name} must be prepared after MXFP8 weight loading")
    if getattr(layer, "_chunk_size", 0) or getattr(layer, "_mxfp8_gmm_nz_layout", False):
        raise ValueError(f"QFM prolog {name} does not accept chunked or grouped-GMM projection weights")
    if getattr(layer, "bias", None) is not None:
        raise ValueError(f"QFM prolog {name} requires a bias-free projection")
    if any(getattr(layer, "mxfp8_tp_padding", (0, 0))):
        raise ValueError(f"QFM prolog {name} does not accept TP-padded MX groups")

    weight, scale = layer.weight, layer.weight_scale
    if weight.dtype != torch.float8_e4m3fn or tuple(weight.shape) != (inputs, outputs):
        raise ValueError(f"QFM prolog {name} expects processed E4M3 weight [{inputs}, {outputs}]")
    expected_scale = (inputs // (MX_GROUP_SIZE * MX_SCALE_PAIR), outputs, MX_SCALE_PAIR)
    if tuple(scale.shape) != expected_scale or scale.dtype not in (torch.uint8, torch.float8_e8m0fnu):
        raise ValueError(f"QFM prolog {name} expects paired E8M0 scales {expected_scale}")
    if weight.device != scale.device:
        raise ValueError(f"QFM prolog {name} weight and scale must share a device")
    # Convert the internal NZ format before slicing/transposing. Tensor shape
    # alone cannot reveal that physical storage is NZ rather than ND.
    weight_nd = torch_npu.npu_format_cast(weight.detach(), ACL_FORMAT_ND)
    private_weight = weight_nd.transpose(0, 1).contiguous().clone()
    private_scale = scale.detach().transpose(0, 1).contiguous().view(torch.float8_e8m0fnu).clone()
    return private_weight, private_scale


def prepare_prolog_weights(impl):
    """Return DSL kwargs without replacing or modifying any original parameter."""
    if getattr(impl, "use_mla_rope", True):
        raise ValueError("QFM prolog requires K3 NoRoPE")
    q_rank, nope_dim = impl.q_lora_rank, impl.qk_nope_head_dim
    if q_rank not in Q_RANKS or nope_dim not in Q_NOPE_DIMS:
        raise ValueError(f"QFM prolog requires q rank in {Q_RANKS} and q_nope dim in {Q_NOPE_DIMS}")
    if impl.kv_lora_rank != KV_RANK or impl.qk_rope_head_dim != RAW_DIM:
        raise ValueError("QFM prolog requires latent512 + raw64")
    heads = impl.num_heads
    if isinstance(heads, bool) or not isinstance(heads, int) or not 1 <= heads <= 128:
        raise ValueError("QFM prolog requires 1..128 TP-local heads")

    norm_qa, norm_kva = impl.q_a_layernorm, impl.kv_a_layernorm
    eps = float(norm_qa.variance_epsilon)
    if not math.isfinite(eps) or eps <= 0 or eps != float(norm_kva.variance_epsilon):
        raise ValueError("QFM prolog requires equal positive finite Q/KV RMSNorm epsilon")
    for name, norm, size in (("q_a", norm_qa, q_rank), ("kv_a", norm_kva, KV_RANK)):
        # Loading-time check only. The native schema has no norm-bias input;
        # zero correction is harmless, nonzero/NaN correction must not vanish.
        bias = getattr(norm, "bias", None)
        if getattr(norm, "bias_loaded", False) and (bias is None or bool(torch.count_nonzero(bias).item())):
            raise ValueError(
                f"QFM DSL Prolog cannot apply loaded nonzero {name} RMSNorm bias; "
                "disable enable_mlapo or use a bias-aware Prolog implementation"
            )
        if norm.weight.dtype not in (torch.bfloat16, torch.float32) or tuple(norm.weight.shape) != (size,):
            raise ValueError(f"QFM prolog {name} RMSNorm weight must be BF16 or FP32 [{size}]")

    fused = getattr(impl, "fused_qkv_a_proj", None)
    first_projection = fused if fused is not None else getattr(impl, "q_a_proj", None)
    hidden_size = getattr(first_projection, "input_size", None)
    if (
        not isinstance(hidden_size, int)
        or not MIN_HIDDEN_SIZE <= hidden_size <= MAX_HIDDEN_SIZE
        or hidden_size % HIDDEN_ALIGNMENT
    ):
        raise ValueError("QFM prolog requires hidden size 1024..8192 aligned to 256")
    if fused is not None:
        weight, scale = _projection(fused, "fused_qkv_a_proj", hidden_size, q_rank + KV_RANK + RAW_DIM)
        wqa, wkva = weight.split((q_rank, KV_RANK + RAW_DIM), dim=0)
        descale_wqa, descale_wkva = scale.split((q_rank, KV_RANK + RAW_DIM), dim=0)
    else:
        q_a = getattr(impl, "q_a_proj", None)
        kv_a = getattr(impl, "kv_a_proj_with_mqa", None)
        if q_a is None or kv_a is None:
            raise ValueError("QFM prolog needs fused_qkv_a_proj or explicit q_a_proj and kv_a_proj_with_mqa")
        wqa, descale_wqa = _projection(q_a, "q_a_proj", hidden_size, q_rank)
        wkva, descale_wkva = _projection(kv_a, "kv_a_proj_with_mqa", hidden_size, KV_RANK + RAW_DIM)

    wqb, descale_wqb = _projection(impl.q_proj, "q_b_proj", q_rank, heads * (nope_dim + RAW_DIM))
    wkb = impl.W_UK_T
    if wkb.dtype != torch.bfloat16 or tuple(wkb.shape) != (heads, nope_dim, KV_RANK):
        raise ValueError("QFM prolog W_UK_T must be BF16 [local_heads, q_nope_dim, 512], without padded/global heads")
    result = {
        "wqa": wqa,
        "wqb": wqb,
        "wkva": wkva,
        "wkb": torch_npu.npu_format_cast(wkb.detach(), ACL_FORMAT_ND).contiguous().clone(),
        "descale_wqa": descale_wqa,
        "descale_wqb": descale_wqb,
        "descale_wkva": descale_wkva,
        "norm_weight_qa": norm_qa.weight.detach().float().contiguous().clone(),
        "norm_weight_kva": norm_kva.weight.detach().float().contiguous().clone(),
        "norm_eps": eps,
    }
    if any(value.device != wqa.device for value in result.values() if isinstance(value, torch.Tensor)):
        raise ValueError("QFM prolog projections and norms must share a device")
    return result

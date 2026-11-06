# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CANN package Prolog frontend for K3's expanded BF16 FIA Prefill.

The package ``cann_ops_transformer.mla_prolog`` wraps MlaPrologV4WeightNz,
whose NoPE interface is documented in ops-transformer v9.2.0, revision
dff221330eecd2132233b1e73ed1072cd8925140. This does not establish that a
particular container has the matching Python wrapper or native runtime.

Native Prolog normally absorbs UK into Q and returns a 512-dimensional Q.
For expanded FIA, a PRIVATE UK=[I_128 | 0] embeds the unabsorbed 128-wide Q
into that output. We keep its first 128 columns and append raw64. This is
not an inverse or pseudoinverse, and never replaces the model's real UK.
The operator still performs its 128-to-512 multiplication, emits the wider
Q, and writes two current-chunk BF16 scratch buffers. Those costs require
an NPU benchmark; fusion alone does not establish a performance benefit.

Only current rows are staged here. The caller writes the persistent full576
FP8 cache and runs FIA; no BF16 history cache or Decode DSL path is changed.
"""

import torch
import torch_npu

NATIVE_HIDDEN_SIZES = (1024, 2048, 3072, 4096, 5120, 6144, 7168, 7680, 8192)
NATIVE_Q_RANKS = (1536, 2048)
Q_NOPE_DIM = 128
RAW_DIM = 64
KV_RANK = 512
CURRENT_BLOCK_SIZE = 128
MX_GROUP_SIZE = 32
ACL_FORMAT_FRACTAL_NZ = 29


def prepare_native_fia_prolog_weights(impl):
    """Prepare private native layouts once, after ModelSlim weight loading."""
    # Reuse the processed-MXFP8 contract checks without importing the linear
    # implementation during attention-module initialization.
    from vllm_ascend.attention.kimi_k3_qfm_prolog_weights import prepare_prolog_weights

    if impl.q_lora_rank not in NATIVE_Q_RANKS or impl.qk_nope_head_dim != Q_NOPE_DIM:
        raise ValueError("Native FIA Prolog requires q rank 1536/2048 and q_nope dim 128")
    for name in ("q_a_layernorm", "kv_a_layernorm"):
        if getattr(impl, name).weight.dtype != torch.bfloat16:
            raise ValueError("Native FIA Prolog requires BF16 RMSNorm weights")
    prepared = prepare_prolog_weights(impl)
    hidden_size = prepared["wqa"].shape[1]
    if hidden_size not in NATIVE_HIDDEN_SIZES:
        raise ValueError(f"Native FIA Prolog hidden size must be in {NATIVE_HIDDEN_SIZES}")

    native = {}
    for native_name, prepared_name in (
        ("weight_dq", "wqa"),
        ("weight_uq_qr", "wqb"),
        ("weight_dkv_kr", "wkva"),
    ):
        # DSL helpers return ND [out,in]. The package takes NZ [in,out].
        native[native_name] = torch_npu.npu_format_cast(
            prepared[prepared_name].transpose(0, 1).contiguous(), ACL_FORMAT_FRACTAL_NZ
        )
    for native_name, prepared_name in (
        ("dequant_scale_w_dq", "descale_wqa"),
        ("dequant_scale_w_uq_qr", "descale_wqb"),
        ("dequant_scale_w_dkv_kr", "descale_wkva"),
    ):
        # ModelSlim pairs successive Group32 E8 codes. Native consumes them
        # flattened as [out,in/32], not the DSL [out,in/64,2] view.
        native[native_name] = prepared[prepared_name].flatten(1).contiguous()
    native["rmsnorm_gamma_cq"] = prepared["norm_weight_qa"].to(torch.bfloat16)
    native["rmsnorm_gamma_ckv"] = prepared["norm_weight_kva"].to(torch.bfloat16)
    native["rmsnorm_epsilon_cq"] = prepared["norm_eps"]
    native["rmsnorm_epsilon_ckv"] = prepared["norm_eps"]

    # No head padding: the V4 interface allows every integer head count 1..128.
    # This matrix is independent of the real UK consumed by Decode and FIA K.
    device = native["weight_dq"].device
    embedded_uk = torch.zeros((impl.num_heads, Q_NOPE_DIM, KV_RANK), dtype=torch.bfloat16, device=device)
    embedded_uk[:, :, :Q_NOPE_DIM] = torch.eye(Q_NOPE_DIM, dtype=torch.bfloat16, device=device)
    native["weight_uk"] = embedded_uk
    return native


class NativeFIAProlog:
    """Return unabsorbed BF16 Q192 and current KV576 without touching history."""

    def __init__(self, impl):
        self.weights = prepare_native_fia_prolog_weights(impl)
        self.num_heads = impl.num_heads
        self.hidden_size = self.weights["weight_dq"].shape[0]
        self.device = self.weights["weight_dq"].device
        # Lazy import keeps package loading in the worker, outside graph capture.
        try:
            from cann_ops_transformer import mla_prolog
        except ImportError as exc:
            raise RuntimeError("Native FIA Prolog needs the CANN package mla_prolog NoPE/V4 wrapper") from exc
        self.op = mla_prolog
        self._capacity = 0
        self._current_latent = None
        self._current_raw = None
        self._indices = None
        scheduler = getattr(getattr(impl, "vllm_config", None), "scheduler_config", None)
        capacity = getattr(scheduler, "max_num_batched_tokens", 0)
        if isinstance(capacity, int) and not isinstance(capacity, bool) and capacity > 0:
            self._reserve(capacity)

    def update_weights(self, impl):
        """Refresh compatible private weights without invalidating addresses."""
        weights = prepare_native_fia_prolog_weights(impl)
        for name, value in weights.items():
            old = self.weights[name]
            if isinstance(value, torch.Tensor):
                if old.shape != value.shape or old.dtype != value.dtype or old.device != value.device:
                    raise ValueError(f"Native FIA Prolog {name} changed; restart workers and recapture")
            elif old != value:
                raise ValueError(f"Native FIA Prolog {name} changed; restart workers and recapture")
        for name, value in weights.items():
            if isinstance(value, torch.Tensor):
                self.weights[name].copy_(value)

    def _reserve(self, tokens):
        if tokens <= self._capacity:
            return
        blocks = (tokens + CURRENT_BLOCK_SIZE - 1) // CURRENT_BLOCK_SIZE
        self._capacity = blocks * CURRENT_BLOCK_SIZE
        self._current_latent = torch.empty(
            (blocks, CURRENT_BLOCK_SIZE, 1, KV_RANK), dtype=torch.bfloat16, device=self.device
        )
        self._current_raw = torch.empty(
            (blocks, CURRENT_BLOCK_SIZE, 1, RAW_DIM), dtype=torch.bfloat16, device=self.device
        )
        self._indices = torch.arange(self._capacity, dtype=torch.int64, device=self.device)

    def __call__(self, hidden):
        if (
            hidden.dtype != torch.bfloat16
            or hidden.ndim != 2
            or hidden.shape[1] != self.hidden_size
            or hidden.device != self.device
        ):
            raise ValueError("Native FIA Prolog needs BF16 [physical_tokens, hidden_dim] on the weight device")
        tokens = hidden.shape[0]
        if tokens == 0:
            return hidden.new_empty((0, self.num_heads, Q_NOPE_DIM + RAW_DIM)), hidden.new_empty((0, KV_RANK + RAW_DIM))
        self._reserve(tokens)
        x, scales = torch_npu.npu_dynamic_mx_quant(hidden.contiguous(), dst_type=torch.float8_e4m3fn, scale_alg=0)
        scales = scales.view(torch.float8_e8m0fnu).reshape(tokens, self.hidden_size // MX_GROUP_SIZE).contiguous()
        blocks = (tokens + CURRENT_BLOCK_SIZE - 1) // CURRENT_BLOCK_SIZE
        latent_cache = self._current_latent[:blocks]
        raw_cache = self._current_raw[:blocks]
        q_latent, q_raw, _, _, _ = self.op(
            token_x=x,
            dequant_scale_x=scales,
            kv_cache=latent_cache,
            kr_cache=raw_cache,
            cache_index=self._indices[:tokens],
            rope_sin=None,
            rope_cos=None,
            cache_mode="PA_BSND",
            weight_quant_mode=3,
            query_quant_mode=0,
            kv_cache_quant_mode=0,
            **self.weights,
        )
        if (
            q_latent.dtype != torch.bfloat16
            or q_raw.dtype != torch.bfloat16
            or tuple(q_latent.shape) != (tokens, self.num_heads, KV_RANK)
            or tuple(q_raw.shape) != (tokens, self.num_heads, RAW_DIM)
        ):
            raise RuntimeError("Native FIA Prolog returned incompatible BF16 Q512/raw64 outputs")
        q = torch.cat((q_latent[:, :, :Q_NOPE_DIM], q_raw), dim=-1)
        current = torch.cat((latent_cache.view(-1, KV_RANK)[:tokens], raw_cache.view(-1, RAW_DIM)[:tokens]), dim=-1)
        return q, current

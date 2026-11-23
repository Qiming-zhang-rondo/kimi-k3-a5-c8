#!/usr/bin/env python3
"""Pure C4 scale experiments on one unchanged, fixed-TopK MLA read path.

The original C4 implementation remains the baseline. Candidates only change
arguments to the installed MXFP4 quantizer; they retain its E2M1/E8M0 bytes,
BF16 RoPE, page mapping, query, sparse indices, and one attention call.
"""
import hashlib

import numpy as np

from kv_window_backend import NPUWindowCase
from kv_window_common import dequantize_mxfp4, paged_array


RECIPES = ("baseline", "alg2_max6", "alg2_max7", "alg2_max7_25")
_MAXIMA = {"alg2_max6": 6.0, "alg2_max7": 7.0, "alg2_max7_25": 7.25}


def _packed_identity(packed, baseline, dim, rope_dim):
    """Hashes and byte changes of logical token records, outside timing."""
    packed, baseline = np.asarray(packed), np.asarray(baseline)
    if packed.dtype != np.uint8 or baseline.dtype != np.uint8 or packed.shape != baseline.shape:
        raise RuntimeError("C4 identity requires equally shaped raw uint8 caches")
    quant_end = dim // 2
    scale_start = quant_end + 2 * rope_dim
    if packed.shape[-1] != scale_start + dim // 32:
        raise RuntimeError("Unexpected packed C4 token width")
    changed = packed != baseline
    return {
        "scope": "logical [B,K,1,packed_width] token records, before page padding",
        "sha256": hashlib.sha256(packed.tobytes()).hexdigest(),
        "baseline_sha256": hashlib.sha256(baseline.tobytes()).hexdigest(),
        "shape": list(packed.shape),
        "total_bytes": int(packed.size),
        "changed_bytes": int(changed.sum()),
        "changed_tokens": int(changed.any(axis=-1).sum()),
        "quant_changed_bytes": int(changed[..., :quant_end].sum()),
        "rope_changed_bytes": int(changed[..., quant_end:scale_start].sum()),
        "scale_changed_bytes": int(changed[..., scale_start:].sum()),
        "identical_to_baseline": not bool(changed.any()),
    }


class C4ScaleBackend:
    """One NPUWindowCase shared by the unchanged controls and C4 candidates.

    ``base`` is injectable for CPU contract tests. Production always constructs
    the existing adapter with compact BF16 indices, as validated on A5.
    """
    def __init__(self, data, device="npu:0", *, base=None):
        self.base = base if base is not None else NPUWindowCase(data, device=device, bf16_indices="compact")
        self.data = self.base.data
        self.environment = dict(self.base.environment)
        self.environment["c4_scale_experiment"] = {
            "window": 0,
            "selection": "unchanged supplied TopK; no QLI call",
            "query_and_rope": "unchanged BF16",
            "reader": "unchanged cloud_kv_quant_sparse_flash_attention",
            "c4_attention_calls_per_read": 1,
            "baseline_quantizer_arguments": {"dst_type": "float4_e2m1fn_x2"},
            "baseline_defaults": "all other quantizer arguments omitted, exactly as the existing C4 adapter",
            "unsupported_recipe_policy": "propagate quantizer error; never substitute another recipe",
        }
        self._variants = {}
        self._logical_kv = self._logical_rope = None

    def host(self, tensor):
        return self.base.host(tensor)

    def baseline(self, kind="bf16"):
        return self.base.call(kind)

    def baseline_run(self, kind="bf16", write=False):
        return self.base.run(kind, write=write)

    def baseline_decoded(self, kind="c4"):
        if kind == "bf16":
            return self.data["kv"].copy()
        return self.base.decoded_kv(kind)

    def prepare(self, recipe_name):
        if recipe_name not in RECIPES:
            raise ValueError(f"Unknown C4 recipe {recipe_name!r}; expected one of {RECIPES}")
        if recipe_name not in self._variants:
            self._variants[recipe_name] = C4ScaleVariant(self, recipe_name)
        return self._variants[recipe_name]

    def _original_logical_inputs(self):
        if self._logical_kv is None:
            self._logical_kv = self.base.tensor(self.data["kv"], self.base.torch.bfloat16)
            self._logical_rope = self.base.tensor(self.data["k_rope"], self.base.torch.bfloat16)
        return self._logical_kv, self._logical_rope


class C4ScaleVariant:
    def __init__(self, backend, recipe_name):
        self.backend, self.base, self.recipe_name = backend, backend.base, recipe_name
        base = self.base
        self._baseline = recipe_name == "baseline"
        self._quantizer_arguments = {"dst_type": base.fp4_type}
        if not self._baseline:
            self._quantizer_arguments.update(axis=-1, round_mode="rint", block_size=32,
                                             scale_alg=2, dst_type_max=_MAXIMA[recipe_name])
        self.metadata = {
            "recipe": recipe_name,
            "quantizer": "torch_npu.npu_dynamic_mx_quant",
            "quantizer_arguments": dict(self._quantizer_arguments, dst_type="float4_e2m1fn_x2"),
            "baseline_reuses_original_cache_and_write": self._baseline,
            "format": "low-nibble-first E2M1; E8M0 scale/32; BF16 RoPE",
            "window": 0,
            "attention_calls_per_read": 1,
            "read_scope": "one unchanged quantized sparse-attention call",
            "write_scope": "current-token quantization + physical-slot write + one unchanged read",
            "prepare_scope": "full original BF16 KV quantization and cache allocation outside timing",
        }
        if self._baseline:
            self.cache = base.c4
            self._decoded = None
            packed_logical = base.c4_logical
        else:
            kv, rope = backend._original_logical_inputs()
            with base.torch.inference_mode():
                packed, quant, scale = self._quantize(kv, rope)
            packed_logical = base.bytes_host(packed).reshape(base.batch, base.keys, 1, -1)
            qbytes = base.bytes_host(quant).reshape(base.batch, base.keys, 1, base.dim // 2)
            sbytes = base.bytes_host(scale).reshape(base.batch, base.keys, 1, base.dim // 32)
            self._decoded = dequantize_mxfp4(qbytes, sbytes, dim=base.dim)
            self.cache = base.tensor(paged_array(packed_logical, base.page, base.table_cpu)).view(base.torch.int8)
        self.packed_identity = _packed_identity(packed_logical, base.c4_logical, base.dim, base.rope_dim)
        if self.packed_identity["rope_changed_bytes"]:
            raise RuntimeError("C4 scale recipe unexpectedly changed BF16 RoPE bytes")
        self.metadata["packed_identity"] = self.packed_identity

    def _quantize(self, kv, rope):
        """Use one recipe for full-cache preparation and current-token writes.

        The byte/shape contract and packing match NPUWindowCase.quantize.
        Unsupported keywords or algorithms propagate directly from Torch-NPU.
        """
        base, t = self.base, self.base.torch
        quant, scale = base.torch_npu.npu_dynamic_mx_quant(
            kv.contiguous().view(-1, 1, base.dim), **self._quantizer_arguments)
        if quant.element_size() != 1 or scale.element_size() != 1:
            raise RuntimeError("MXFP4 quantizer must expose packed FP4 and E8M0 as one-byte storage")
        count, quant_bytes, scale_bytes = kv.numel() // base.dim, base.dim // 2, base.dim // 32
        if quant.numel() * quant.element_size() != count * quant_bytes or scale.numel() * scale.element_size() != count * scale_bytes:
            raise RuntimeError(f"Unexpected c4 quantizer output sizes: q={tuple(quant.shape)}, scale={tuple(scale.shape)}")
        rope_bytes = rope.contiguous().view(t.uint8).reshape(count, 2 * base.rope_dim)
        packed = t.cat((quant.contiguous().view(t.uint8).reshape(count, quant_bytes), rope_bytes,
                        scale.contiguous().view(t.uint8).reshape(count, scale_bytes)), -1)
        return packed, quant, scale

    def decoded_kv(self):
        return self.base.decoded_kv("c4") if self._baseline else self._decoded.copy()

    def call(self):
        if self._baseline:
            return self.base.call("c4")
        return self.base._read("c4", self.base.indices, cache=self.cache, statistics=False)[0]

    def run(self, write=False):
        if self._baseline:
            return self.base.run("c4", write=write)
        if write:
            packed, _, _ = self._quantize(self.base.current_kv, self.base.current_rope)
            self.cache.view(self.base.torch.uint8).view(-1, self.cache.shape[-1]).index_copy_(
                0, self.base.current_slots, packed)
        return self.call()

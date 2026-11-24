# SFA Cold-State Merge — AscendC Kernel Source

## What this is

Source modification for `sparse_flash_attention` (AscendC, arch35/Ascend950DT)
that fuses cold-end state (output/max/sum) into the BF16 hot-end attention
kernel's final output stage, eliminating the external Triton merge kernel.

## Key results driving this work (A5, 2026-11-23)

| Config | pipeline µs | W0 ratio | L2 vs BF16 |
|---|---|---|---|
| C4 W0 (baseline) | 87.5 | 1.000 | 0.10410 |
| BF16 hot + Triton merge (two-pass) | 126.2 | 1.438 | 0.09901 |
| C8 hot + Triton merge (two-pass) | 128.6 | 1.466 | 0.09908 |
| hot_read_merge (Triton fused) | 284.4 | 3.258 | 0.09908 |

The external Triton merge costs ~14 µs. Fusing it into the native kernel
targets the gap between 126 µs and the ~87 µs W0 floor.

## Files

```
arch35/
  cold_state_merge.h                        ← core merge algorithm (new file)
  sparse_flash_attention_service_vector_mla.h  ← integration point (ProcessVec2)
  sparse_flash_attention_kernel_mla.h          ← kernel entry (pass params)
  sparse_flash_attention_kvcache.h             ← shared constants
  sparse_flash_attention_common_arch35.h        ← arch35 common
  sparse_flash_attention_service_cube_mla.h     ← cube service (unchanged)
sparse_flash_attention.cpp                     ← kernel entry point
sparse_flash_attention_common.h                ← ConstInfo extension
sparse_flash_attention_template_tiling_key.h    ← tiling keys
INTEGRATION.md                                  ← detailed integration guide
README.md                                       ← this file
```

## Math contract

Hot output `O_h` is already normalized (`O_h = A_h / s_h`).
Cold output `O_c` is also normalized (`O_c = A_c / s_c`).

Merge (algebraically identical to combining unnormalized accumulators):
```
m   = max(m_h, m_c)
w_h = exp(m_h - m)        // stable, ≤ 1.0
w_c = exp(m_c - m)        // stable, ≤ 1.0
Z   = s_h * w_h + s_c * w_c
O   = (O_h * s_h * w_h + O_c * s_c * w_c) / Z
```

Edge cases:
- Hot only (cold_present=false): O = O_h (no-op)
- Cold only (hot_present=false): O = O_c
- Both empty: O = 0
- Non-finite stats on valid side: propagate as error (do not mask)

## Build

This operator is part of the `custom_cot` CANN vendor package.
Modified files are drop-in replacements for:
```
custom_cot/op_impl/ai_core/tbe/custom_cot_impl/ascendc/sparse_flash_attention/
```

Rebuild requires the full CANN custom operator build pipeline (not included
in the experiment container). See `INTEGRATION.md` for the exact build steps.

## Status

- [x] Core merge algorithm written and reviewed
- [x] Integration points identified (ProcessVec2 L743-L744)
- [x] Math contract verified against kv_native_merge.py reference
- [ ] Kernel compilation (requires CANN build pipeline)
- [ ] On-device precision verification
- [ ] Paired performance measurement (10-round AB/BA)
- [ ] ACLNN API layer + tiling integration

## Reference

- Experiment framework: Qiming-zhang-rondo/kimi-k3-a5-c8
- Container: c4 on 10.83.89.178 (CANN 9.2.1, Ascend950DT)
- Last verified: 2026-11-23, bf16_hot_store_partition 126.17 µs (1.438x W0)

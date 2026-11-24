"""Formula-only Torch checks; does not compile or execute cold_state_merge.h."""
import torch
import torch_npu

torch.manual_seed(42)
device = "npu:0"
print("SCOPE: PyTorch formulas only; no AscendC header/kernel execution.")

B, Q, H, D = 1, 4, 64, 512

print("=" * 60)
print("Test 1: Normal case (both hot and cold have data)")
print("=" * 60)

for trial in range(5):
    hot_out = torch.randn(B, Q, H, D, dtype=torch.bfloat16, device=device)
    cold_out = torch.randn(B, Q, H, D, dtype=torch.bfloat16, device=device)
    hot_max = torch.randn(B, Q, H, 1, dtype=torch.float32, device=device) * 5
    cold_max = torch.randn(B, Q, H, 1, dtype=torch.float32, device=device) * 5
    hot_sum = torch.rand(B, Q, H, 1, dtype=torch.float32, device=device) * 100 + 1
    cold_sum = torch.rand(B, Q, H, 1, dtype=torch.float32, device=device) * 100 + 1

    # Reference: standard LSE merge
    m = torch.maximum(hot_max, cold_max)
    w_h = torch.exp(hot_max - m)
    w_c = torch.exp(cold_max - m)
    Z = hot_sum * w_h + cold_sum * w_c
    ref = (hot_out.float() * hot_sum * w_h + cold_out.float() * cold_sum * w_c) / Z

    # Mine: cold_state_merge.h formula
    combinedMax = torch.maximum(hot_max, cold_max)
    wH = torch.exp(hot_max - combinedMax)
    wC = torch.exp(cold_max - combinedMax)
    z = hot_sum * wH + cold_sum * wC
    zh = hot_sum * wH / z
    zc = cold_sum * wC / z
    mine = hot_out.float() * zh + cold_out.float() * zc

    diff = (ref - mine).abs().max().item()
    rel = diff / ref.abs().max().item() if ref.abs().max() > 0 else 0
    ok = "✅" if diff < 1e-5 else "❌"
    print(f"  trial {trial}: max_diff={diff:.2e}, rel={rel:.2e} {ok}")
    assert diff < 1e-5, f"formula mismatch: {diff}"

print()
print("=" * 60)
print("Test 2: Numerical stability (extreme values)")
print("=" * 60)

for trial in range(5):
    hot_out = torch.randn(B, Q, H, D, dtype=torch.bfloat16, device=device) * 100
    cold_out = torch.randn(B, Q, H, D, dtype=torch.bfloat16, device=device) * 100
    hot_max = torch.full((B, Q, H, 1), 50.0, device=device)
    cold_max = torch.full((B, Q, H, 1), -50.0, device=device)
    hot_sum = torch.full((B, Q, H, 1), 1000.0, device=device)
    cold_sum = torch.full((B, Q, H, 1), 0.001, device=device)

    combinedMax = torch.maximum(hot_max, cold_max)
    wH = torch.exp(hot_max - combinedMax)
    wC = torch.exp(cold_max - combinedMax)
    z = hot_sum * wH + cold_sum * wC
    zh = hot_sum * wH / z
    zc = cold_sum * wC / z
    result = hot_out.float() * zh + cold_out.float() * zc
    expected = hot_out.float()

    diff = (result - expected).abs().max().item()
    finite = torch.isfinite(result).all().item()
    ok = "✅" if finite and diff < 1e-3 else "❌"
    print(f"  trial {trial}: extreme(max=50 vs -50): diff={diff:.2e}, finite={finite} {ok}")
    assert finite and diff < 1e-3, f"extreme formula mismatch: {diff}"

print()
print("=" * 60)
print("Test 3: Split softmax == merged (ground truth equivalence)")
print("=" * 60)

n_hot, n_cold, n_q, n_h, d = 128, 1920, 4, 8, 64

for trial in range(5):
    hot_scores = torch.randn(n_q, n_h, n_hot, dtype=torch.float32, device=device)
    cold_scores = torch.randn(n_q, n_h, n_cold, dtype=torch.float32, device=device)
    hot_vals = torch.randn(n_q, n_h, n_hot, d, device=device)
    cold_vals = torch.randn(n_q, n_h, n_cold, d, device=device)

    # Full softmax (ground truth)
    all_scores = torch.cat([hot_scores, cold_scores], dim=-1)
    all_vals = torch.cat([hot_vals, cold_vals], dim=-2)
    probs = torch.softmax(all_scores, dim=-1)
    ref = torch.einsum("qhk,qhkd->qhd", probs, all_vals)

    # Split + my merge
    hot_probs = torch.softmax(hot_scores, dim=-1)
    cold_probs = torch.softmax(cold_scores, dim=-1)
    hot_out = torch.einsum("qhk,qhkd->qhd", hot_probs, hot_vals)
    cold_out = torch.einsum("qhk,qhkd->qhd", cold_probs, cold_vals)

    hot_max = hot_scores.max(dim=-1, keepdim=True).values
    cold_max = cold_scores.max(dim=-1, keepdim=True).values
    hot_sum = torch.exp(hot_scores - hot_max).sum(dim=-1, keepdim=True)
    cold_sum = torch.exp(cold_scores - cold_max).sum(dim=-1, keepdim=True)

    m = torch.maximum(hot_max, cold_max)
    wH = torch.exp(hot_max - m)
    wC = torch.exp(cold_max - m)
    z = hot_sum * wH + cold_sum * wC
    merged = hot_out * (hot_sum * wH / z) + cold_out * (cold_sum * wC / z)

    diff = (merged - ref).abs().max().item()
    rel = diff / ref.abs().max().item() if ref.abs().max() > 0 else 0
    ok = "✅" if rel < 1e-4 else "❌"
    print(f"  trial {trial}: full_softmax vs split+merge: abs={diff:.4e}, rel={rel:.4e} {ok}")
    assert rel < 1e-4, f"split softmax formula mismatch: {rel}"

print()
print("=" * 60)
print("Presence/invalid-state tests are in test_uploaded_merge_regressions.py")
print("=" * 60)
print("  This script does not execute presence branches or the C++ header.")

print()
print("=" * 60)
print("TORCH FORMULA CHECKS PASSED; native kernel compilation/runtime untested")
print("=" * 60)


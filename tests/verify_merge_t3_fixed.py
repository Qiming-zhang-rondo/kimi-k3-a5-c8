"""Test 3 fixed: correct unnormalized sums."""
import torch
import torch_npu

torch.manual_seed(42)
device = "npu:0"

n_hot, n_cold, n_q, n_h, d = 128, 1920, 4, 8, 64

print("Test 3 (fixed): Split softmax == merged")
print(f"  shape: hot={n_hot}, cold={n_cold}, Q={n_q}, H={n_h}, D={d}")

for trial in range(5):
    hot_scores = torch.randn(n_q, n_h, n_hot, dtype=torch.float32, device=device)
    cold_scores = torch.randn(n_q, n_h, n_cold, dtype=torch.float32, device=device)
    hot_vals = torch.randn(n_q, n_h, n_hot, d, device=device)
    cold_vals = torch.randn(n_q, n_h, n_cold, d, device=device)

    # Reference: full attention computed in one pass (no concat, save memory)
    all_max = torch.maximum(
        hot_scores.max(dim=-1, keepdim=True).values,
        cold_scores.max(dim=-1, keepdim=True).values
    )
    exp_hot = torch.exp(hot_scores - all_max)
    exp_cold = torch.exp(cold_scores - all_max)
    num = torch.einsum("qhk,qhkd->qhd", exp_hot, hot_vals) + \
          torch.einsum("qhk,qhkd->qhd", exp_cold, cold_vals)
    den = exp_hot.sum(dim=-1, keepdim=True) + exp_cold.sum(dim=-1, keepdim=True)
    ref = num / den

    # Split: each side does its own softmax (with own max)
    hot_max = hot_scores.max(dim=-1, keepdim=True).values   # [Q,H,1]
    cold_max = cold_scores.max(dim=-1, keepdim=True).values # [Q,H,1]

    # UNNORMALIZED sums (not softmax output sums which are 1.0)
    hot_sum = torch.exp(hot_scores - hot_max).sum(dim=-1, keepdim=True)  # [Q,H,1]
    cold_sum = torch.exp(cold_scores - cold_max).sum(dim=-1, keepdim=True) # [Q,H,1]

    # Normalized outputs (what the kernel produces)
    hot_out = torch.einsum("qhk,qhkd->qhd",
        torch.softmax(hot_scores, dim=-1), hot_vals)
    cold_out = torch.einsum("qhk,qhkd->qhd",
        torch.softmax(cold_scores, dim=-1), cold_vals)

    # My merge (cold_state_merge.h formula)
    m = torch.maximum(hot_max, cold_max)
    wH = torch.exp(hot_max - m)
    wC = torch.exp(cold_max - m)
    z = hot_sum * wH + cold_sum * wC
    merged = hot_out * (hot_sum * wH / z) + cold_out * (cold_sum * wC / z)

    diff = (merged - ref).abs().max().item()
    rel = diff / ref.abs().max().item() if ref.abs().max() > 0 else 0
    ok = "✅" if rel < 1e-4 else "❌"
    print(f"  trial {trial}: abs={diff:.4e}, rel={rel:.4e} {ok}")

print("\nDONE")

"""CPU smoke test for gptq_quantize_layer.
Verifies:
  1. Output shapes and dtypes match the amax path.
  2. On Hessian-weighted reconstruction error (||X(W-Wq)||^2), GPTQ beats amax.
  3. Dead column handling doesn't crash.

Usage: python3 dev/test_gptq_cpu.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

# Import just the functions we need, bypassing CUDA init.
import importlib.util
spec = importlib.util.spec_from_file_location("tg", "train_gpt.py")
tg_module = importlib.util.module_from_spec(spec)
# The module-level code only defines things, doesn't run main — safe to import on CPU.
spec.loader.exec_module(tg_module)

gptq_quantize_layer = tg_module.gptq_quantize_layer
quantize_float_tensor = tg_module.quantize_float_tensor


def hessian_weighted_err(W: torch.Tensor, Wq: torch.Tensor, X: torch.Tensor) -> float:
    """||X @ (W - Wq).T||_F^2 — the loss GPTQ actually minimizes."""
    diff = W - Wq
    return float(torch.sum((X @ diff.T) ** 2).item())


def test_basic_shapes():
    torch.manual_seed(0)
    out, in_dim = 32, 64
    W = torch.randn(out, in_dim) * 0.1
    X = torch.randn(128, in_dim)
    H = X.T @ X
    q, s = gptq_quantize_layer(W, H, bits=4)
    assert q.shape == W.shape, f"q shape {q.shape} vs W shape {W.shape}"
    assert q.dtype == torch.int8, f"q dtype is {q.dtype}"
    assert s.shape == (out,), f"scale shape {s.shape}"
    assert s.dtype == torch.float16, f"scale dtype {s.dtype}"
    # Reconstruct
    Wq = q.float() * s.float().unsqueeze(1)
    # No NaN/Inf
    assert torch.isfinite(Wq).all()
    # Basic accuracy: at least within the quantization noise floor
    assert (W - Wq).abs().max().item() < 0.05, "GPTQ reconstruction way off"
    print(f"[PASS] test_basic_shapes: max_abs_err={((W - Wq).abs().max().item()):.4f}")


def test_beats_amax_on_weighted_err():
    """GPTQ should beat amax on Hessian-weighted error when H is not a scaled identity."""
    torch.manual_seed(1)
    out, in_dim = 64, 128
    W = torch.randn(out, in_dim) * 0.2
    # Non-uniform activation distribution: columns have very different scales.
    col_scales = torch.exp(torch.linspace(-2, 2, in_dim))  # 0.13x to 7.4x range
    X = torch.randn(256, in_dim) * col_scales
    H = X.T @ X

    # GPTQ path
    q_g, s_g = gptq_quantize_layer(W, H, bits=4, damp_percent=0.01)
    Wq_g = q_g.float() * s_g.float().unsqueeze(1)

    # amax path
    q_a, s_a = quantize_float_tensor(W, bits=4, use_amax=True)
    Wq_a = q_a.float() * s_a.float().view(-1, 1)

    err_g = hessian_weighted_err(W, Wq_g, X)
    err_a = hessian_weighted_err(W, Wq_a, X)
    ratio = err_g / err_a
    print(f"[INFO] weighted_err: gptq={err_g:.2f} amax={err_a:.2f} ratio={ratio:.3f}")
    assert ratio < 0.85, f"GPTQ should beat amax by at least 15% on weighted err, got ratio={ratio:.3f}"
    print(f"[PASS] test_beats_amax_on_weighted_err (GPTQ {(1-ratio)*100:.1f}% lower error)")


def test_dead_columns():
    """Rows of X that are all zero yield dead columns in H — should not crash."""
    torch.manual_seed(2)
    out, in_dim = 16, 32
    W = torch.randn(out, in_dim) * 0.1
    X = torch.randn(64, in_dim)
    X[:, 5] = 0.0  # dead column
    X[:, 17] = 0.0
    H = X.T @ X
    q, s = gptq_quantize_layer(W, H, bits=6)
    Wq = q.float() * s.float().unsqueeze(1)
    assert torch.isfinite(Wq).all(), "dead-column path produced NaN/Inf"
    # Dead columns should be quantized to 0
    assert Wq[:, 5].abs().max().item() == 0.0, "dead column should be zeroed"
    assert Wq[:, 17].abs().max().item() == 0.0, "dead column should be zeroed"
    print(f"[PASS] test_dead_columns")


def test_ill_conditioned_hessian():
    """Near-singular H — damping must kick in."""
    torch.manual_seed(3)
    out, in_dim = 16, 32
    W = torch.randn(out, in_dim) * 0.1
    # Low-rank X: only 4 samples
    X = torch.randn(4, in_dim)
    H = X.T @ X  # rank 4, very singular
    q, s = gptq_quantize_layer(W, H, bits=6, damp_percent=0.1)
    Wq = q.float() * s.float().unsqueeze(1)
    assert torch.isfinite(Wq).all(), "ill-conditioned H produced non-finite output"
    print(f"[PASS] test_ill_conditioned_hessian")


if __name__ == "__main__":
    test_basic_shapes()
    test_beats_amax_on_weighted_err()
    test_dead_columns()
    test_ill_conditioned_hessian()
    print("\nAll GPTQ CPU tests passed.")

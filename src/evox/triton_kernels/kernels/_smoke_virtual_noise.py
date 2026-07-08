"""Standalone smoke test for the Triton virtual-noise kernels.

Verifies that the previously-crashing forward config (the one that triggered a
Triton ``OutOfResources`` shared-memory error) now runs cleanly on CUDA, plus a
few other tile sizes for robustness.

Run with::

    uv run python src/evox/triton_kernels/kernels/_smoke_virtual_noise.py
"""

import torch

from evox.triton_kernels.kernels.virtual_noise import (
    virtual_bias_gradient,
    virtual_perturbed_linear,
    virtual_weight_gradient,
)


def _build_case(out_features, in_features, batch, pop_size, sigma=0.05, offset=0, device="cuda"):
    """Build one (x, weight, bias, seeds) configuration on `device`."""
    g = torch.Generator(device=device).manual_seed(123)
    weight = torch.randn(out_features, in_features, device=device, generator=g) * 0.1
    bias = torch.randn(out_features, device=device, generator=g) * 0.1
    x = torch.randn(batch, in_features, device=device, generator=g)
    seeds = (torch.arange(1, pop_size + 1, device=device) * 7919).to(torch.int32)
    return x, weight, bias, seeds


def _run_one(out_features, in_features, batch, pop_size, sigma=0.05, offset=0):
    """Run forward + gradients on CUDA and check invariants. Returns True on success."""
    print(
        f"\n  config: weight=({out_features},{in_features}), batch={batch}, "
        f"pop_size={pop_size}, sigma={sigma}, offset={offset}"
    )

    try:
        x, weight, bias, seeds = _build_case(
            out_features, in_features, batch, pop_size, sigma=sigma, offset=offset
        )

        # --- Forward (Triton path on CUDA) ---
        y1 = virtual_perturbed_linear(x, weight, bias, seeds, sigma, offset)
        if not torch.isfinite(y1).all():
            print("    FAIL: forward output has NaN/Inf")
            return False
        print(f"    forward ok: out={tuple(y1.shape)}, finite=True, mean={y1.mean().item():.4f}")

        # --- Determinism: re-running forward gives identical results ---
        y2 = virtual_perturbed_linear(x, weight, bias, seeds, sigma, offset)
        if not torch.equal(y1, y2):
            max_abs = (y1 - y2).abs().max().item()
            print(f"    FAIL: forward is not deterministic (max abs diff={max_abs:.3e})")
            return False
        print("    determinism ok: re-run identical")

        # --- Weight gradient (Triton path on CUDA) ---
        fitness = torch.randn(pop_size, device="cuda", dtype=torch.float32)
        wg = virtual_weight_gradient(fitness, seeds, [out_features, in_features], sigma, pop_size, offset)
        if not torch.isfinite(wg).all():
            print("    FAIL: weight gradient has NaN/Inf")
            return False
        print(f"    weight grad ok: shape={tuple(wg.shape)}, finite=True")

        # --- Bias gradient (Triton path on CUDA) ---
        bg = virtual_bias_gradient(
            fitness, seeds, [out_features], sigma, pop_size, offset + out_features * in_features
        )
        if not torch.isfinite(bg).all():
            print("    FAIL: bias gradient has NaN/Inf")
            return False
        print(f"    bias grad ok: shape={tuple(bg.shape)}, finite=True")

        return True
    except Exception as e:  # noqa: BLE001
        print(f"    EXCEPTION ({type(e).__name__}): {e}")
        return False


def main():
    if not torch.cuda.is_available():
        print("CUDA not available — skipping (smoke test requires CUDA to exercise the Triton path).")
        return 0

    props = torch.cuda.get_device_properties(0)
    print(f"Device: {props.name}, compute capability {props.major}.{props.minor}")
    print(f"  shared_memory_per_block = {getattr(props, 'shared_memory_per_block', 'N/A')}")

    # The CRASHING config from benchmark_virtual_lora_es.py:
    #   Linear(d_ff=512, d_model=128) => weight.shape = (out=128, in=512)
    #   pop_size=16 (the first population size, which crashed at init_step),
    #   batch ~ N_SAMPLES=1024.
    configs = [
        # (out_features, in_features, batch, pop_size) — the failing config FIRST.
        (128, 512, 1024, 16),
        # Other tile sizes for robustness.
        (64, 64, 256, 8),
        (512, 128, 512, 16),
        (256, 1024, 256, 16),
        (128, 512, 1024, 32),
    ]

    all_ok = True
    for cfg in configs:
        ok = _run_one(*cfg)
        all_ok = all_ok and ok
        status = "PASS" if ok else "FAIL"
        print(f"  -> {status}")

    print("\n" + "=" * 60)
    if all_ok:
        print("ALL CONFIGS PASSED (no OutOfResources, all finite, deterministic).")
        return 0
    else:
        print("ONE OR MORE CONFIGS FAILED.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

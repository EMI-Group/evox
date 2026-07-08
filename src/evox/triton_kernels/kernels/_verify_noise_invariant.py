"""Corrected gold-standard verification of the forward<->gradient noise invariant.

Extracts the weight noise N_i[out=j, in=k] from the Triton FORWARD output
(with W=0, b=0, sigma=1, X=I) and compares it to the noise consumed by the
Triton GRADIENT kernel (via a direct linear extraction from the gradient).

Key indexing (this is where the previous attempt had a transpose bug):
  Forward: Y[i, b, j] = sum_k X[b,k] N_i[j,k] + nb_i[j].
  With X = I, W = 0, b = 0, sigma = 1:
      Y[i, b, j] = N_i[j, b] + nb_i[j]
  =>  N_i[j, k] = Y[i, k, j] - nb_i[j]     (note: b becomes the 'k' / in index)
  So N_extracted has shape (pop, out, in) = y_eye.transpose(1,2) - nb[:, :, None].

The gradient kernel: grad[j,k] = sum_i fitness_i * N_i[j,k] / (pop * sigma).
So if the invariant holds, grad_triton must equal einsum("i,ijk->jk", fitness, N_extracted)/(p*sigma).

Run with::

    uv run python src/evox/triton_kernels/kernels/_verify_noise_invariant.py
"""

import torch

from evox.triton_kernels.kernels.virtual_noise import (
    virtual_bias_gradient,
    virtual_perturbed_linear,
    virtual_weight_gradient,
)


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available — skipping (requires Triton path on CUDA).")
        return 0

    device = "cuda"
    n = 64  # square: out_features = in_features = batch = n
    pop_size = 8
    sigma = 1.0
    offset = 0

    g = torch.Generator(device=device).manual_seed(7)
    weight0 = torch.zeros(n, n, device=device)  # zero weight -> only noise contributes
    bias0 = torch.zeros(n, device=device)
    seeds = (torch.arange(1, pop_size + 1, device=device) * 131).to(torch.int32)

    # Identity input.
    x_eye = torch.eye(n, device=device, dtype=torch.float32)
    # Zero input: isolates the bias noise term (broadcast over all batch rows).
    x_zero = torch.zeros(n, n, device=device, dtype=torch.float32)

    y_eye = virtual_perturbed_linear(x_eye, weight0, bias0, seeds, sigma, offset)
    y_zero = virtual_perturbed_linear(x_zero, weight0, bias0, seeds, sigma, offset)

    nb = y_zero[:, 0, :]                                # (pop, out) = nb_i[j]
    # N_extracted[i, out=j, in=k] = Y[i, k, j] - nb_i[j]
    N = y_eye.transpose(1, 2) - nb[:, :, None]          # (pop, out, in)

    # --- Verify weight gradient against the extracted noise ---
    fitness = torch.randn(pop_size, device=device, dtype=torch.float32)
    grad_ref = torch.einsum("i,ijk->jk", fitness, N) / (pop_size * sigma)  # (out, in)
    grad_triton = virtual_weight_gradient(fitness, seeds, [n, n], sigma, pop_size, offset)

    w_err = (grad_ref - grad_triton).abs().max().item()
    w_rel = w_err / (grad_ref.abs().max().item() + 1e-12)
    print(f"weight gradient: max abs err = {w_err:.3e}, relative = {w_rel:.3e}")

    # --- Verify bias gradient against the extracted bias noise ---
    bias_offset = offset + n * n
    bg_ref = torch.einsum("i,ij->j", fitness, nb) / (pop_size * sigma)
    bg_triton = virtual_bias_gradient(fitness, seeds, [n], sigma, pop_size, bias_offset)
    b_err = (bg_ref - bg_triton).abs().max().item()
    b_rel = b_err / (bg_ref.abs().max().item() + 1e-12)
    print(f"bias   gradient: max abs err = {b_err:.3e}, relative = {b_rel:.3e}")

    # fp32 reduction-order round-off: expect ~1e-5 abs.
    tol = 1e-3
    ok = (w_rel < tol) and (b_rel < tol)
    print("=" * 60)
    if ok:
        print(f"NOISE INVARIANT HOLDS: forward<->gradient noise consistent (rel err < {tol}).")
        return 0
    else:
        print(f"NOISE INVARIANT VIOLATED: relative error exceeds {tol}.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Tests for the virtual (never-materialized) noise kernels in
:mod:`evox.triton_kernels.kernels.virtual_noise`.

These tests exercise the **Triton** code path on CUDA (where Triton is
available and a GPU is present) and guard against regressions in:

- **Resource safety**: the fused forward kernel
  (:func:`virtual_perturbed_linear`) must not raise a Triton
  ``OutOfResources`` (shared-memory) error for large reduction dimensions
  (regression test for a bug where ``Linear(512, 128)`` with ``batch=1024``
  exceeded the per-block shared-memory limit).
- **Correctness / noise invariant**: the forward kernel and the gradient
  kernels (:func:`virtual_weight_gradient` / :func:`virtual_bias_gradient`)
  must regenerate the *same* per-individual Gaussian noise for a given
  ``(seed, offset, element_index)`` triple, so the analytical gradient is a
  valid estimate of the forward perturbation.

The tests are skipped automatically when CUDA is unavailable (e.g. on CI
without a GPU), since the Triton kernels only dispatch on CUDA devices. The
pure-PyTorch fallback path is covered by the CPU-only tests elsewhere.
"""

import unittest

import torch

from evox.triton_kernels.kernels.virtual_noise import (
    virtual_bias_gradient,
    virtual_perturbed_linear,
    virtual_weight_gradient,
)


def _cuda_available() -> bool:
    return torch.cuda.is_available()


@unittest.skipUnless(_cuda_available(), "CUDA required to exercise the Triton path")
class TestVirtualNoiseKernelResources(unittest.TestCase):
    """Regression tests for the shared-memory ``OutOfResources`` crash.

    The forward kernel fuses a per-individual Gaussian-noise perturbation into
    a ``tl.dot`` matmul. Originally it loaded the *entire* inner (reduction)
    dimension in one tile (``BLOCK_IN`` up to 512), which on shared-memory
    constrained GPUs (e.g. sm_86, ~99 KB opt-in limit) exceeded the per-block
    budget and crashed. The kernel now tiles the reduction dimension and
    selects an adaptive software-pipelining depth (``num_stages``).
    """

    def setUp(self):
        torch.manual_seed(42)

    def _run_forward(
        self,
        out_features,
        in_features,
        batch,
        pop_size,
        sigma=0.05,
        offset=0,
    ):
        """Run the fused forward kernel and assert it is finite & deterministic."""
        device = "cuda"
        g = torch.Generator(device=device).manual_seed(123)
        weight = torch.randn(out_features, in_features, device=device, generator=g) * 0.1
        bias = torch.randn(out_features, device=device, generator=g) * 0.1
        x = torch.randn(batch, in_features, device=device, generator=g)
        seeds = (torch.arange(1, pop_size + 1, device=device) * 7919).to(torch.int32)

        y1 = virtual_perturbed_linear(x, weight, bias, seeds, sigma, offset)
        self.assertEqual(tuple(y1.shape), (pop_size, batch, out_features))
        self.assertTrue(torch.isfinite(y1).all())

        # Re-running must be bit-identical (deterministic PRNG).
        y2 = virtual_perturbed_linear(x, weight, bias, seeds, sigma, offset)
        self.assertTrue(torch.equal(y1, y2))
        return y1, seeds, sigma, offset

    def test_crashing_benchmark_config(self):
        """The exact config that triggered the original crash.

        ``benchmark_virtual_lora_es.py`` builds a transformer with a
        ``Linear(d_ff=512, d_model=128)`` layer (weight shape ``(128, 512)``)
        and runs with ``batch=1024`` and ``pop_size=16``.
        """
        self._run_forward(out_features=128, in_features=512, batch=1024, pop_size=16)

    def test_large_reduction_dim(self):
        """A wide inner dimension (1024) exercises the reduction tiling."""
        self._run_forward(out_features=256, in_features=1024, batch=256, pop_size=16)

    def test_small_square(self):
        """Small square config as a baseline."""
        self._run_forward(out_features=64, in_features=64, batch=256, pop_size=8)

    def test_gradient_kernels_run(self):
        """Weight & bias gradient kernels must run without ``OutOfResources``."""
        out_features, in_features, pop_size = 128, 512, 16
        offset = 0
        device = "cuda"
        seeds = (torch.arange(1, pop_size + 1, device=device) * 7919).to(torch.int32)
        sigma = 0.05
        fitness = torch.randn(pop_size, device=device, dtype=torch.float32)

        wg = virtual_weight_gradient(
            fitness, seeds, [out_features, in_features], sigma, pop_size, offset
        )
        self.assertEqual(tuple(wg.shape), (out_features, in_features))
        self.assertTrue(torch.isfinite(wg).all())

        bias_offset = offset + out_features * in_features
        bg = virtual_bias_gradient(
            fitness, seeds, [out_features], sigma, pop_size, bias_offset
        )
        self.assertEqual(tuple(bg.shape), (out_features,))
        self.assertTrue(torch.isfinite(bg).all())


@unittest.skipUnless(_cuda_available(), "CUDA required to exercise the Triton path")
class TestVirtualNoiseInvariant(unittest.TestCase):
    """Verify the forward<->gradient noise regeneration invariant.

    The forward kernel perturbs ``W`` with noise ``N_i[out=j, in=k]`` and the
    gradient kernel regenerates the *same* ``N_i`` to compute the analytical
    gradient estimate. They must agree (within fp32 round-off), otherwise the
    ES gradient estimate would be wrong.

    Strategy: with ``W=0``, ``b=0``, ``sigma=1`` and the identity input
    ``X=I``, the forward output isolates the noise exactly::

        Y[i, b, j] = sum_k X[b,k] * N_i[j,k] + nb_i[j]
                   = N_i[j, b] + nb_i[j]          (X = I)

    so ``N_i[j, k] = Y[i, k, j] - nb_i[j]``. We then compare the gradient
    kernel's output to ``einsum("i,ijk->jk", fitness, N) / (pop * sigma)``.
    """

    def test_forward_gradient_noise_consistency(self):
        device = "cuda"
        n = 64  # square: out_features = in_features = batch = n
        pop_size = 8
        sigma = 1.0
        offset = 0

        weight0 = torch.zeros(n, n, device=device)
        bias0 = torch.zeros(n, device=device)
        seeds = (torch.arange(1, pop_size + 1, device=device) * 131).to(torch.int32)

        x_eye = torch.eye(n, device=device, dtype=torch.float32)
        x_zero = torch.zeros(n, n, device=device, dtype=torch.float32)

        y_eye = virtual_perturbed_linear(x_eye, weight0, bias0, seeds, sigma, offset)
        y_zero = virtual_perturbed_linear(x_zero, weight0, bias0, seeds, sigma, offset)

        nb = y_zero[:, 0, :]  # (pop, out) = nb_i[j]
        # N_extracted[i, out=j, in=k] = Y[i, k, j] - nb_i[j]
        n_extracted = y_eye.transpose(1, 2) - nb[:, :, None]  # (pop, out, in)

        fitness = torch.randn(pop_size, device=device, dtype=torch.float32)

        # Weight gradient must match the noise extracted from the forward pass.
        grad_ref = torch.einsum("i,ijk->jk", fitness, n_extracted) / (pop_size * sigma)
        grad_triton = virtual_weight_gradient(
            fitness, seeds, [n, n], sigma, pop_size, offset
        )
        w_err = (grad_ref - grad_triton).abs().max().item()
        w_rel = w_err / (grad_ref.abs().max().item() + 1e-12)
        self.assertLess(w_rel, 1e-3, f"weight gradient noise invariant violated (rel={w_rel:.3e})")

        # Bias gradient must match the bias noise extracted from the forward pass.
        bias_offset = offset + n * n
        bg_ref = torch.einsum("i,ij->j", fitness, nb) / (pop_size * sigma)
        bg_triton = virtual_bias_gradient(fitness, seeds, [n], sigma, pop_size, bias_offset)
        b_err = (bg_ref - bg_triton).abs().max().item()
        b_rel = b_err / (bg_ref.abs().max().item() + 1e-12)
        self.assertLess(b_rel, 1e-3, f"bias gradient noise invariant violated (rel={b_rel:.3e})")


if __name__ == "__main__":
    unittest.main()

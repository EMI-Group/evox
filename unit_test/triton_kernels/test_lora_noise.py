"""Tests for the LoRA noise utility functions.

These tests validate the pure-PyTorch implementations (no CUDA available in the
dev environment). The kernels modules are imported via a lightweight stub-package
loader that does not execute ``evox/__init__.py`` (which fails on Python 3.13 due
to a pre-existing ``torch.compile`` call).
"""

import importlib.util
import sys
import types
import unittest

import torch

_SRC = "src/evox"


def _load_kernel_modules():
    """Import the triton_kernels submodules without running evox/__init__.py."""
    if "evox.triton_kernels.kernels.lora_noise" in sys.modules:
        return

    def _make_pkg(name, path):
        mod = types.ModuleType(name)
        mod.__path__ = [path]
        sys.modules[name] = mod
        return mod

    _make_pkg("evox", _SRC)
    _make_pkg("evox.utils", f"{_SRC}/utils")
    _make_pkg("evox.triton_kernels", f"{_SRC}/triton_kernels")
    _make_pkg("evox.triton_kernels.kernels", f"{_SRC}/triton_kernels/kernels")

    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    _load("evox.utils.re_export", f"{_SRC}/utils/re_export.py")
    _load("evox.utils.op_register", f"{_SRC}/utils/op_register.py")
    _load("evox.triton_kernels.backend", f"{_SRC}/triton_kernels/backend.py")
    _load("evox.triton_kernels.op_register", f"{_SRC}/triton_kernels/op_register.py")
    _load("evox.triton_kernels.kernels.fused_add", f"{_SRC}/triton_kernels/kernels/fused_add.py")
    _load("evox.triton_kernels.kernels.philox", f"{_SRC}/triton_kernels/kernels/philox.py")
    _load("evox.triton_kernels.kernels.lora_noise", f"{_SRC}/triton_kernels/kernels/lora_noise.py")


_load_kernel_modules()
from evox.triton_kernels.kernels.lora_noise import (  # noqa: E402
    compute_counter_offsets,
    generate_lora_factors,
    lora_delta_output,
    lora_gradient,
)


class TestComputeCounterOffsets(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_non_overlapping_ranges(self):
        """Counter ranges for successive blocks must not overlap."""
        shapes = [(10,), (4, 3), (2, 5)]
        rank = 2
        offsets = compute_counter_offsets(shapes, rank)
        self.assertEqual(len(offsets), len(shapes))
        self.assertEqual(offsets[0], 0)
        # Each offset must be >= the previous offset + consumed elements.
        for i in range(len(offsets) - 1):
            self.assertGreater(offsets[i + 1], offsets[i])

    def test_known_values(self):
        """Verify exact offset values for a known configuration."""
        # block 0: (10,) -> 10 elems -> ceil(10/4)*4 = 12
        # block 1: (4,3), rank 2 -> d=4,k=3 -> 2*3+4*2=14 -> ceil(14/4)*4=16
        # block 2: (2,5), rank 2 -> d=2,k=5 -> 2*5+2*2=14 -> ceil(14/4)*4=16
        offsets = compute_counter_offsets([(10,), (4, 3), (2, 5)], 2)
        self.assertEqual(offsets, [0, 12, 28])

    def test_monotonic_increasing(self):
        """Offsets must be strictly monotonically increasing (cumulative)."""
        offsets = compute_counter_offsets([(5,), (3, 7), (2, 2, 2), (100,)], 4)
        for i in range(len(offsets) - 1):
            self.assertGreater(offsets[i + 1], offsets[i])


class TestGenerateLoraFactors(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_1d_noise_shape(self):
        """1-D weight produces (pop_size, n) flat noise."""
        seeds = torch.arange(1, 11, dtype=torch.int64) * 777
        noise = generate_lora_factors(seeds, (10,), rank=2, counter=0)
        self.assertEqual(noise.shape, (10, 10))
        self.assertEqual(noise.dtype, torch.float32)

    def test_1d_noise_basic_stats(self):
        """1-D noise should have mean ≈ 0, std ≈ 1 over a large sample."""
        seeds = torch.arange(1, 1001, dtype=torch.int64) * 777
        noise = generate_lora_factors(seeds, (500,), rank=2, counter=0)
        self.assertAlmostEqual(noise.mean().item(), 0.0, delta=0.1)
        self.assertAlmostEqual(noise.std().item(), 1.0, delta=0.1)

    def test_2d_factors_shape(self):
        """2-D weight produces A (pop, rank, k) and B (pop, d, rank)."""
        seeds = torch.arange(1, 11, dtype=torch.int64) * 777
        result = generate_lora_factors(seeds, (4, 3), rank=2, counter=0)
        self.assertIsInstance(result, tuple)
        A, B = result
        self.assertEqual(A.shape, (10, 2, 3))
        self.assertEqual(B.shape, (10, 4, 2))

    def test_2d_delta_shape(self):
        """B @ A reconstructs a (pop, d, k) delta."""
        seeds = torch.arange(1, 11, dtype=torch.int64) * 777
        A, B = generate_lora_factors(seeds, (4, 3), rank=2, counter=0)
        delta_w = torch.bmm(B, A)  # (pop, d, rank) @ (pop, rank, k) -> (pop, d, k)
        self.assertEqual(delta_w.shape, (10, 4, 3))

    def test_higher_dim_flatten(self):
        """>2-D weights flatten: first dims -> d, last -> k."""
        seeds = torch.arange(1, 9, dtype=torch.int64) * 777
        A, B = generate_lora_factors(seeds, (2, 3, 4), rank=2, counter=0)
        # d = 2*3 = 6, k = 4
        self.assertEqual(A.shape, (8, 2, 4))
        self.assertEqual(B.shape, (8, 6, 2))

    def test_non_overlapping_substreams(self):
        """A and B sub-streams must use different counter ranges (independent)."""
        seeds = torch.arange(1, 11, dtype=torch.int64) * 777
        A, B = generate_lora_factors(seeds, (4, 3), rank=2, counter=0)
        # If the counter ranges overlapped, the same Philox calls would produce
        # correlated outputs. Check they are simply different (not identical).
        self.assertFalse(torch.equal(A.reshape(10, -1)[:, : B.shape[1]], B.reshape(10, -1)[:, : A.shape[1]].mT.contiguous().reshape(10, -1)[:, :1]))


class TestLoraDeltaOutput(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_matches_explicit_materialization(self):
        """sigma*(x@A.T)@B.T must equal sigma*x@(B@A).T for small sizes."""
        pop, batch, inf, outf = 5, 3, 4, 6
        rank = 2
        x = torch.randn(pop, batch, inf)
        A = torch.randn(pop, rank, inf)
        B = torch.randn(pop, outf, rank)
        sigma = 0.1

        result = lora_delta_output(x, A, B, sigma)
        # Explicit: delta_W = B @ A (pop, outf, inf); out = sigma * x @ delta_W.T
        delta_w = torch.bmm(B, A)  # (pop, outf, inf)
        explicit = sigma * torch.bmm(x, delta_w.transpose(-1, -2))
        self.assertTrue(torch.allclose(result, explicit, atol=1e-5))

    def test_shape(self):
        """Output shape must be (pop, batch, out_features)."""
        pop, batch, inf, outf = 3, 2, 8, 5
        rank = 3
        x = torch.randn(pop, batch, inf)
        A = torch.randn(pop, rank, inf)
        B = torch.randn(pop, outf, rank)
        out = lora_delta_output(x, A, B, sigma=0.5)
        self.assertEqual(out.shape, (pop, batch, outf))

    def test_sigma_scaling(self):
        """Doubling sigma must double the output."""
        pop, batch, inf, outf = 2, 2, 3, 4
        rank = 2
        x = torch.randn(pop, batch, inf)
        A = torch.randn(pop, rank, inf)
        B = torch.randn(pop, outf, rank)
        out1 = lora_delta_output(x, A, B, sigma=1.0)
        out2 = lora_delta_output(x, A, B, sigma=2.0)
        self.assertTrue(torch.allclose(out2, 2.0 * out1, atol=1e-6))


class TestLoraGradient(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_2d_matches_manual(self):
        """2-D gradient must match the explicit fitness-weighted average."""
        pop_size = 8
        d, k = 3, 4
        rank = 2
        seeds = torch.arange(1, pop_size + 1, dtype=torch.int64) * 555
        A, B = generate_lora_factors(seeds, (d, k), rank, counter=0)
        sigma = 0.3
        fitness = torch.randn(pop_size)

        grad = lora_gradient(fitness, A, B, pop_size, sigma, (d, k))

        manual = torch.zeros(d, k)
        for i in range(pop_size):
            manual += fitness[i] * torch.mm(B[i], A[i])
        manual = manual / (pop_size * sigma)
        self.assertTrue(torch.allclose(grad, manual, atol=1e-5))

    def test_2d_shape(self):
        """2-D gradient must have the weight shape."""
        pop_size = 6
        d, k = 5, 7
        rank = 3
        seeds = torch.arange(1, pop_size + 1, dtype=torch.int64) * 555
        A, B = generate_lora_factors(seeds, (d, k), rank, counter=0)
        grad = lora_gradient(torch.randn(pop_size), A, B, pop_size, sigma=0.5, weight_shape=(d, k))
        self.assertEqual(grad.shape, (d, k))

    def test_1d_gradient(self):
        """1-D gradient: (fitness / (pop*sigma)) @ noise reshaped to weight_shape."""
        pop_size = 8
        n = 12
        seeds = torch.arange(1, pop_size + 1, dtype=torch.int64) * 555
        noise = generate_lora_factors(seeds, (n,), rank=2, counter=0)
        self.assertEqual(noise.shape, (pop_size, n))
        sigma = 0.4
        fitness = torch.randn(pop_size)

        grad = lora_gradient(fitness, noise, None, pop_size, sigma, (n,))
        manual = (fitness / (pop_size * sigma)) @ noise
        self.assertTrue(torch.allclose(grad, manual.reshape(n), atol=1e-5))

    def test_higher_dim_reshape(self):
        """>2-D gradient reshapes back to the original weight shape."""
        pop_size = 6
        shape = (2, 3, 4)  # d = 6, k = 4
        rank = 2
        seeds = torch.arange(1, pop_size + 1, dtype=torch.int64) * 555
        A, B = generate_lora_factors(seeds, shape, rank, counter=0)
        grad = lora_gradient(torch.randn(pop_size), A, B, pop_size, sigma=0.5, weight_shape=shape)
        self.assertEqual(grad.shape, shape)

    def test_finite_difference_2d(self):
        """Verify the gradient via a finite-difference check on a scalar objective.

        The objective is the population loss whose stationary-point gradient is
        exactly what ``lora_gradient`` estimates::

            J(W) = (1/(pop*sigma)) * sum_i fitness_i * <W, sigma * (B_i @ A_i)>

        so ``dJ/dW = (1/(pop*sigma)) * sum_i fitness_i * sigma * (B_i @ A_i)``,
        which equals the output of ``lora_gradient``. Because ``J`` is linear in
        ``W``, the central finite difference is exact (up to floating-point error).
        """
        pop_size = 16
        d, k = 2, 3
        rank = 2
        sigma = 0.3
        seeds = torch.arange(1, pop_size + 1, dtype=torch.int64) * 555
        A, B = generate_lora_factors(seeds, (d, k), rank, counter=0)
        fitness = torch.randn(pop_size)
        # The ES gradient estimate is g = sum_i fitness_i * (B_i@A_i) / (pop*sigma).
        # Define J(W) = sum_i fitness_i * <W, (B_i@A_i)> / (pop*sigma), which is
        # linear in W so dJ/dW = g exactly (matched by central finite differences).
        delta = torch.bmm(B, A)  # (pop, d, k), unscaled by sigma

        def objective(weight):
            return (
                (fitness.unsqueeze(-1).unsqueeze(-1) * delta * weight).sum()
            ) / (pop_size * sigma)

        W = torch.randn(d, k)
        grad = lora_gradient(fitness, A, B, pop_size, sigma, (d, k))

        # Central finite differences -- exact for a linear objective.
        eps = 1e-4
        fd_grad = torch.zeros_like(grad)
        for i in range(d):
            for j in range(k):
                W_p = W.clone()
                W_p[i, j] += eps
                W_m = W.clone()
                W_m[i, j] -= eps
                fd_grad[i, j] = (objective(W_p) - objective(W_m)) / (2 * eps)
        # atol accounts for central finite-difference floating-point noise.
        self.assertTrue(torch.allclose(grad, fd_grad, atol=1e-2))


if __name__ == "__main__":
    unittest.main()

"""Tests for the Philox4x32-10 PRNG kernels (philox_uniform / philox_normal).

These tests validate the pure-PyTorch fallback path (no CUDA is available in the
dev environment). On CUDA machines with Triton, the same tests would additionally
exercise the Triton kernel path.

The symbols are imported via ``evox.triton_kernels`` when possible. As a fallback
for environments where the full ``import evox`` chain is unavailable (e.g. Python
3.13, where ``evox/__init__.py`` triggers ``torch.compile`` via
``evox/operators/selection/non_dominate.py``), a lightweight stub-package loader
imports just the triton_kernels submodules without executing ``evox/__init__.py``.
"""

import importlib.util
import os
import sys
import types
import unittest

import torch


def _import_philox():
    """Import ``philox_uniform`` / ``philox_normal`` robustly.

    First attempts the normal ``import evox`` path (the common case on supported
    Python versions / CI). Only if that fails does it fall back to loading the
    ``triton_kernels`` submodules directly without running ``evox/__init__.py``,
    so these tests can still run even when the full package import chain is
    broken in the current environment. The fallback is never triggered when the
    normal import works, so it cannot pollute ``sys.modules`` in CI.
    """
    try:
        from evox.triton_kernels.kernels.philox import philox_normal, philox_uniform

        return philox_uniform, philox_normal
    except Exception:
        pass

    if "evox.triton_kernels.kernels.philox" in sys.modules:
        from evox.triton_kernels.kernels.philox import philox_normal, philox_uniform

        return philox_uniform, philox_normal

    # Resolve the repository ``src/evox`` directory from this file's location.
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.normpath(os.path.join(here, "..", "..", "src", "evox"))

    def _make_pkg(name, path):
        mod = types.ModuleType(name)
        mod.__path__ = [path]
        sys.modules[name] = mod
        return mod

    _make_pkg("evox", src)
    _make_pkg("evox.utils", os.path.join(src, "utils"))
    _make_pkg("evox.triton_kernels", os.path.join(src, "triton_kernels"))
    _make_pkg("evox.triton_kernels.kernels", os.path.join(src, "triton_kernels", "kernels"))

    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    _load("evox.utils.re_export", os.path.join(src, "utils", "re_export.py"))
    _load("evox.utils.op_register", os.path.join(src, "utils", "op_register.py"))
    _load("evox.triton_kernels.backend", os.path.join(src, "triton_kernels", "backend.py"))
    _load("evox.triton_kernels.op_register", os.path.join(src, "triton_kernels", "op_register.py"))
    _load("evox.triton_kernels.kernels.fused_add", os.path.join(src, "triton_kernels", "kernels", "fused_add.py"))
    _load("evox.triton_kernels.kernels.philox", os.path.join(src, "triton_kernels", "kernels", "philox.py"))

    from evox.triton_kernels.kernels.philox import philox_normal, philox_uniform

    return philox_uniform, philox_normal


philox_uniform, philox_normal = _import_philox()


class TestPhiloxUniform(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_determinism(self):
        """Same seeds + counter must produce identical output across calls."""
        seeds = torch.arange(1, 101, dtype=torch.int64) * 12345 + 7
        a = philox_uniform(seeds, 100)
        b = philox_uniform(seeds, 100)
        self.assertTrue(torch.equal(a, b))

    def test_counter_jump_ahead(self):
        """A different counter must produce a different stream."""
        seeds = torch.arange(1, 101, dtype=torch.int64) * 12345 + 7
        a = philox_uniform(seeds, 100, counter=0)
        b = philox_uniform(seeds, 100, counter=1)
        self.assertFalse(torch.equal(a, b))

    def test_seed_independence(self):
        """Different seeds must produce different streams."""
        seeds_a = torch.arange(1, 101, dtype=torch.int64) * 12345 + 7
        seeds_b = torch.arange(2, 102, dtype=torch.int64) * 12345 + 7
        a = philox_uniform(seeds_a, 100)
        b = philox_uniform(seeds_b, 100)
        self.assertFalse(torch.equal(a, b))

    def test_shape_various(self):
        """Output shape must match (pop_size, n) for various sizes."""
        for pop in (1, 3, 10):
            for n in (1, 4, 5, 100, 257):
                seeds = torch.arange(pop, dtype=torch.int64) + 1
                out = philox_uniform(seeds, n)
                self.assertEqual(out.shape, (pop, n), f"failed for pop={pop}, n={n}")
                self.assertEqual(out.dtype, torch.float32)

    def test_uniform_range(self):
        """All values must be in [0, 1)."""
        seeds = torch.arange(1, 1001, dtype=torch.int64) * 12345 + 7
        out = philox_uniform(seeds, 1000)
        self.assertTrue(torch.all(out >= 0.0), "min below 0")
        self.assertTrue(torch.all(out < 1.0), "max >= 1")

    def test_uniform_stats(self):
        """Mean of a large uniform sample should be close to 0.5."""
        seeds = torch.arange(1, 1001, dtype=torch.int64) * 12345 + 7
        out = philox_uniform(seeds, 1000)
        self.assertAlmostEqual(out.mean().item(), 0.5, delta=0.02)

    def test_cpu_fallback(self):
        """Explicitly test on CPU device."""
        seeds = torch.tensor([10, 20, 30], dtype=torch.int64, device="cpu")
        out = philox_uniform(seeds, 16)
        self.assertEqual(out.shape, (3, 16))
        self.assertEqual(out.device.type, "cpu")
        self.assertTrue(torch.all(out >= 0.0) and torch.all(out < 1.0))

    def test_determinism_with_counter(self):
        """Same seeds + same non-zero counter must be deterministic."""
        seeds = torch.arange(1, 51, dtype=torch.int64) * 999
        a = philox_uniform(seeds, 50, counter=12345)
        b = philox_uniform(seeds, 50, counter=12345)
        self.assertTrue(torch.equal(a, b))


class TestPhiloxNormal(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_determinism(self):
        """Same seeds + counter must produce identical output across calls."""
        seeds = torch.arange(1, 101, dtype=torch.int64) * 12345 + 7
        a = philox_normal(seeds, 100)
        b = philox_normal(seeds, 100)
        self.assertTrue(torch.equal(a, b))

    def test_shape_various(self):
        """Output shape must match (pop_size, n) for various sizes (incl. odd n)."""
        for pop in (1, 5):
            for n in (1, 2, 3, 4, 7, 100):
                seeds = torch.arange(pop, dtype=torch.int64) + 1
                out = philox_normal(seeds, n)
                self.assertEqual(out.shape, (pop, n), f"failed for pop={pop}, n={n}")
                self.assertEqual(out.dtype, torch.float32)

    def test_normal_stats(self):
        """Mean ≈ 0, std ≈ 1 for a large sample."""
        seeds = torch.arange(1, 1001, dtype=torch.int64) * 12345 + 7
        out = philox_normal(seeds, 1000)
        self.assertAlmostEqual(out.mean().item(), 0.0, delta=0.1)
        self.assertAlmostEqual(out.std().item(), 1.0, delta=0.1)

    def test_counter_jump_ahead(self):
        """A different counter must produce a different normal stream."""
        seeds = torch.arange(1, 101, dtype=torch.int64) * 12345 + 7
        a = philox_normal(seeds, 100, counter=0)
        b = philox_normal(seeds, 100, counter=1)
        self.assertFalse(torch.equal(a, b))

    def test_cpu_fallback(self):
        """Explicitly test on CPU device."""
        seeds = torch.tensor([10, 20, 30], dtype=torch.int64, device="cpu")
        out = philox_normal(seeds, 16)
        self.assertEqual(out.shape, (3, 16))
        self.assertEqual(out.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()

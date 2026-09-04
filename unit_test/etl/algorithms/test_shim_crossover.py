"""Tests for the canonical crossover operators in ``evox_etl.operators.crossover``.

Converted from the deprecated ``evox_etl.algorithms._shim_crossover`` compat
stub (which re-exported the canonical operators and pinned the sampled index of
``DE_differential_sum`` to int32). Now imports the canonical operators directly
and asserts the canonical dtypes.
"""
import sys
from pathlib import Path
sys.path[0:0] = [str(Path(__file__).resolve().parents[3]), str(Path(__file__).resolve().parents[3] / "src")]

import numpy as np
import etl
from etl import core

from evox_etl.operators.crossover import (
    DE_arithmetic_recombination,
    DE_binary_crossover,
    DE_differential_sum,
    DE_exponential_crossover,
    simulated_binary,
    simulated_binary_half,
)

POP = 6
DIM = 4
PAD = 5
HALF = 4


def _exercise(key, x, num_diff_vects, index, population, mutation, current, CR, K):
    out_sbx = simulated_binary(key, x)
    out_half = simulated_binary_half(key, x)
    ds_r, first_r = DE_differential_sum(key, PAD, num_diff_vects, index, population, F=0.8, replace=True)
    ds_nr, first_nr = DE_differential_sum(key, PAD, num_diff_vects, index, population, F=None, replace=False)
    ds_f, _ = DE_differential_sum(key, PAD, num_diff_vects, index, population, F=0.8, replace=False)
    out_bin = DE_binary_crossover(key, mutation, current, CR)
    out_exp = DE_exponential_crossover(key, mutation, current, CR)
    out_arith = DE_arithmetic_recombination(mutation, current, K)
    return out_sbx, out_half, ds_r, first_r, ds_nr, first_nr, ds_f, out_bin, out_exp, out_arith


class TestCrossoverOperators:
    @classmethod
    def setup_class(cls):
        specs = [
            core.TensorSpec(shape=(), dtype=np.dtype("int64")),
            core.TensorSpec(shape=(2 * HALF, DIM), dtype=np.dtype("float32")),
            core.TensorSpec(shape=(POP,), dtype=np.dtype("int32")),
            core.TensorSpec(shape=(POP,), dtype=np.dtype("int32")),
            core.TensorSpec(shape=(POP, DIM), dtype=np.dtype("float32")),
            core.TensorSpec(shape=(POP, DIM), dtype=np.dtype("float32")),
            core.TensorSpec(shape=(POP, DIM), dtype=np.dtype("float32")),
            core.TensorSpec(shape=(POP,), dtype=np.dtype("float32")),
            core.TensorSpec(shape=(POP,), dtype=np.dtype("float32")),
        ]
        cls.exe = etl.build(_exercise, *specs, backend="numpy")

        rng = np.random.default_rng(0)
        cls.key = np.array(42, dtype=np.int64)
        cls.x = rng.standard_normal((2 * HALF, DIM)).astype(np.float32)
        cls.num_diff_vects = np.ones(POP, dtype=np.int32)
        cls.index = np.full(POP, 3, dtype=np.int32)
        cls.population = rng.standard_normal((POP, DIM)).astype(np.float32)
        cls.mutation = rng.standard_normal((POP, DIM)).astype(np.float32)
        cls.current = rng.standard_normal((POP, DIM)).astype(np.float32)
        cls.CR = np.full(POP, 0.5, dtype=np.float32)
        cls.K = np.linspace(0.1, 0.9, POP).astype(np.float32)

    def run(self):
        outs = etl.run(
            self.exe, self.key, self.x, self.num_diff_vects, self.index,
            self.population, self.mutation, self.current, self.CR, self.K,
        )
        return [t.numpy() for t in outs]

    def test_sbx_shapes_dtypes_and_finiteness(self):
        sbx, half = self.run()[0:2]
        assert sbx.shape == (2 * HALF, DIM) and sbx.dtype == np.float32
        assert half.shape == (HALF, DIM) and half.dtype == np.float32
        assert np.isfinite(sbx).all() and np.isfinite(half).all()

    def test_sbx_changes_some_entries_only(self):
        sbx, half = self.run()[0:2]
        assert not np.array_equal(sbx, self.x)
        assert np.count_nonzero(sbx != self.x) < sbx.size
        p1, p2 = self.x[:HALF], self.x[HALF:]
        assert not np.array_equal(half, p1) and not np.array_equal(half, p2)

    def test_de_differential_sum_shapes_and_replace_semantics(self):
        ds_r, first_r, ds_nr, first_nr, ds_f = self.run()[2:7]
        for ds in (ds_r, ds_nr, ds_f):
            assert ds.shape == (POP, DIM) and ds.dtype == np.float32
            assert np.isfinite(ds).all()
        assert first_r.shape == (POP,) and first_r.dtype == np.int32
        # canonical: with replace=False the fix-up `etl.select(rand_indices == index,
        # pop_size - 1, rand_indices)` promotes the int32 draws to int64 via the
        # python-int scalar (torch promotes the same way) — the old shim cast this
        # back to int32; the canonical operator does not.
        assert first_nr.shape == (POP,) and first_nr.dtype == np.int64
        assert (first_r >= 0).all() and (first_r < POP).all()
        assert (first_nr != 3).all()
        assert np.array_equal(ds_f, ds_nr * np.float32(0.8))

    def test_de_crossovers_pick_from_either_parent(self):
        *_, out_bin, out_exp, _ = self.run()
        for out in (out_bin, out_exp):
            assert out.shape == (POP, DIM) and out.dtype == np.float32
            assert np.isfinite(out).all()
            assert np.all((out == self.mutation) | (out == self.current))
        assert np.all(np.count_nonzero(out_bin != self.current, axis=1) >= 1)

    def test_de_arithmetic_recombination_exact_value(self):
        out_arith = self.run()[-1]
        expected = self.current + self.K[:, None] * (self.mutation - self.current)
        assert out_arith.shape == (POP, DIM) and out_arith.dtype == np.float32
        np.testing.assert_allclose(out_arith, expected, rtol=0, atol=1e-6)

    def test_determinism_same_key_twice(self):
        first = self.run()
        second = self.run()
        for a, b in zip(first, second):
            assert np.array_equal(a, b)

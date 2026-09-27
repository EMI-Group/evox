"""Parity tests: functional evox_etl ES-variant step protocol vs torch evox.

This is the only torch-importing file in this suite; mirrors
``src/evox_etl/metrics/tests/parity/test_parity.py``.

The driver below is deliberately self-contained: it drives a functional
step-protocol module (``init(config, key)`` / ``step(config, state,
evaluate)``) through ``etl.build`` / ``etl.run`` on the ``numpy`` backend, so
this package suite never depends on the sibling ``unit_test/etl`` tree (that
tree is read only as the reference for the ``etl.build``/``etl.run`` contract:
scalar graph inputs are 0-d int64 arrays, state pytrees are passed as specs,
and ``etl.run`` returns concrete tensors that must be read with ``.numpy()``).

Noise injection is what makes the two sides comparable at all: the etl port
and the torch reference otherwise draw from different RNG streams.  Both are
patched to the SAME fixed noise — ``torch.randn`` through the module attribute
and ``etl.random.normal`` through an in-graph ``etl.ops.constant`` (the
``evox_etl`` ES modules do ``import etl.random as random``, so patching that
attribute affects the trace; ``etl.ops.constant`` rejects a raw ndarray, hence
the explicit ``etl.core.tensor``).
"""

from dataclasses import replace
from typing import Any, Callable

import numpy as np
import pytest
import torch

import etl
import etl.random as random
from etl.core import TensorSpec

from evox.algorithms import ARS, ASEBO, CMAES

import evox_etl.algorithms.so.es_variants.ars as ars
import evox_etl.algorithms.so.es_variants.asebo as asebo
import evox_etl.algorithms.so.es_variants.cma_es as cma_es

F32 = np.dtype("float32")
INT64 = np.dtype("int64")


# --------------------------------------------------------------------------- #
# drivers / helpers
# --------------------------------------------------------------------------- #
def _spec_tree(tree: Any) -> Any:
    """Replace every tensor leaf of a state pytree with a matching TensorSpec."""
    return etl.tree_map(
        lambda t: TensorSpec(shape=tuple(t.shape), dtype=np.dtype(t.dtype)), tree
    )


def _array(value: Any) -> np.ndarray:
    """ETL/torch tensor (or ndarray) -> numpy array."""
    return np.asarray(value.numpy() if hasattr(value, "numpy") else value)


def _max_abs_diff(a: Any, b: Any) -> float:
    """Maximum absolute difference between two tensors/arrays, in float64."""
    return float(
        np.max(np.abs(_array(a).astype(np.float64) - _array(b).astype(np.float64)))
    )


def _fixed_noise(seed: int, shape: tuple[int, ...]) -> np.ndarray:
    """Deterministic float32 noise tensor injected into BOTH implementations."""
    return np.random.default_rng(seed).normal(size=shape).astype(np.float32)


def _inject_fixed_noise(monkeypatch: pytest.MonkeyPatch, values: np.ndarray) -> None:
    """Force both RNG entry points to return ``values`` (see module docstring)."""
    fixed = np.asarray(values, dtype=np.float32)

    def torch_randn(*shape: Any, **kwargs: Any) -> torch.Tensor:
        return torch.from_numpy(fixed.copy())

    def etl_normal(key, shape, mean=0.0, std=1.0, dtype=None):
        return etl.ops.constant(etl.core.tensor(fixed.copy()))

    monkeypatch.setattr(torch, "randn", torch_randn)
    monkeypatch.setattr(random, "normal", etl_normal)


def _sphere(candidates: Any) -> Any:
    """Toy fitness evaluated INSIDE the trace: ``f(x) = sum(x * x)`` per row."""
    return etl.sum(candidates * candidates, axes=1)


def _drive(
    mod: Any,
    config: Any,
    n_gens: int,
    *,
    state_transform: Callable[[Any], Any] | None = None,
    key_seed: int = 0,
) -> Any:
    """Drive a step-protocol module for ``n_gens`` generations; return the state.

    The config is baked into both executables as a closure constant and the
    step executable is built once and re-run (every ES state here keeps
    constant tensor-leaf shapes).  ``state_transform`` may rewrite the incoming
    state inside the trace.
    """

    def generation(state: Any) -> Any:
        if state_transform is not None:
            state = state_transform(state)
        return mod.step(config, state, _sphere)

    init_exe = etl.build(
        lambda key: mod.init(config, key), TensorSpec((), INT64), backend="numpy"
    )
    state = etl.run(init_exe, np.asarray(key_seed, dtype=np.int64))
    step_exe = etl.build(generation, _spec_tree(state), backend="numpy")
    for _ in range(n_gens):
        state = etl.run(step_exe, state)
    return state


def _asebo_subspace_input(dim: int, half: int, seed: int = 0) -> np.ndarray:
    """Build a well-posed ``X`` for the ASEBO SVD/sign block (see test 1 (b)).

    Returns ``X = U diag(s) V^T`` (float32) with ``s[-1] = 0`` and the two
    choices LAPACK leaves undetermined for a zero singular value neutralised by
    construction:

    * ``U[:, -1]`` is the constant vector — forced, since ``X``'s columns sum to
      (up to float32) zero, i.e. ``1`` is a left null vector;
    * ``V[:, -1]`` (the right null vector) is supported on the first ``half``
      rows, so the null direction contributes nothing to ``V``'s ``[half:]``
      block;
    * ``V[-1, :]`` is the ``e[-2]`` coordinate vector, so the arbitrary sign
      pattern ``Vt * signs`` applies to that last data row can only flip its
      single non-zero entry (an outer product is invariant under that).
    """
    rng = np.random.default_rng(seed)
    const = np.full(dim, 1.0 / np.sqrt(dim))
    v_null = np.zeros(dim)
    v_null[:half] = rng.normal(size=half)
    v_null /= np.linalg.norm(v_null)
    e_last = np.zeros(dim)
    e_last[-1] = 1.0
    block = rng.normal(size=(dim, dim - 2))
    for vec in (e_last, v_null):  # orthogonal complement of {e_last, v_null}
        block -= np.outer(vec, vec @ block)
    v_basis, _ = np.linalg.qr(block)
    V = np.empty((dim, dim))
    V[:, : dim - 2] = v_basis
    V[:, dim - 2] = e_last
    V[:, dim - 1] = v_null
    core = rng.normal(size=(dim, dim - 1))
    core -= np.outer(const, const @ core)
    u_basis, _ = np.linalg.qr(core)
    U = np.empty((dim, dim))
    U[:, : dim - 1] = u_basis
    U[:, dim - 1] = const
    s = np.append(np.linspace(3.0, 0.6, dim - 1), 0.0)
    return ((U * s) @ V.T).astype(F32)


def _torch_asebo_subspace_block(
    X: np.ndarray, half: int, *, transpose: bool = True, sign_matrix: bool = True
) -> np.ndarray:
    """Replicate ``src/evox/algorithms/so/es_variants/asebo.py`` lines 95-107.

    CONVENTION FACT (do not "fix"): ``torch.svd(X, some=True)``'s THIRD output
    is **V** (== ``Vh.T``), not ``Vh`` — deprecated-API convention; the port
    transposes ``etl.svd``'s ``Vh`` to match (commit 16543cd5).  Using
    ``torch.linalg.svd`` here would test the wrong target.

    ``transpose=False`` (the non-transposed ``Vh[half:].T @ Vh[half:]`` slicing)
    and ``sign_matrix=False`` (a length-k sign VECTOR) reproduce the two
    pre-fix formulations of the port, for the discriminating-power guard.
    """
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    U, _S, Vt = torch.svd(Xt, some=True)
    max_abs_cols = torch.argmax(torch.abs(U), dim=0)
    if sign_matrix:
        signs = torch.sign(U[max_abs_cols, :])  # (k, k) sign MATRIX
    else:
        signs = torch.sign(U[max_abs_cols, torch.arange(U.shape[1])])  # (k,) vector
    U = U * signs
    Vt = Vt * signs
    Vh = Vt.T  # the pre-fix port fed etl.svd's Vh straight into the slicing
    U_ort = Vt[half:] if transpose else Vh[half:]
    return (U_ort.T @ U_ort).numpy()


# --------------------------------------------------------------------------- #
# 1. ASEBO
# --------------------------------------------------------------------------- #
def test_asebo_state_parity_with_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """ASEBO: end-to-end state parity + the SVD/sign subspace block vs torch."""
    dim, pop_size, half = 10, 8, 4
    center0 = np.full((dim,), 5.0, dtype=np.float32)
    # asebo samples `random.normal(subkey, (dim, half))` (half = pop_size // 2).
    # Noise seed 3 keeps the (float32-cancellation-limited) end-to-end spread of
    # (a) at |dcenter| = 1.9e-6 / |dgrad_subspace| = 1.7e-5, i.e. ~6x inside the
    # 1e-4 assertion; the spread is dominated by the mirrored fitness difference
    # of a Sphere population of magnitude ~125 evaluated in float32 on each side.
    _inject_fixed_noise(monkeypatch, _fixed_noise(3, (dim, half)))

    torch_alg = ASEBO(
        pop_size=pop_size, center_init=torch.full((dim,), 5.0), sigma=0.5, lr=1.0
    )
    torch_alg.evaluate = lambda p: (p * p).sum(-1)
    torch_alg.step()
    torch_alg.step()

    config = asebo.make_asebo(
        pop_size=pop_size, center_init=center0, sigma=0.5, lr=1.0
    )
    state = _drive(asebo, config, 2)

    # (a) end-to-end parity.  For every generation with gen_counter <=
    # subspace_dims the SVD/sign block is INERT: UUT is masked to zero and
    # alpha is forced to 1.0, so `cov = sigma * (alpha / dim) * I` is isotropic
    # and the whole step is comparable (not just the inert block).
    assert _max_abs_diff(state.center, torch_alg.center) <= 1e-4
    assert _max_abs_diff(state.sigma, torch_alg.sigma) <= 1e-4
    assert _max_abs_diff(state.grad_subspace, torch_alg.grad_subspace) <= 1e-4

    # (b) the stored subspace factor `UUT_ort` (`UUT` itself is masked to zeros
    # while gen_counter <= subspace_dims, so `UUT_ort` is the discriminating
    # stored quantity).  X is replaced by a constructed, well-posed
    # grad_subspace — the port's own step, same code path, explicit input.
    #
    # X = grad_subspace - colmean(grad_subspace) must be SQUARE: the reference
    # does `U * signs` with `U` (m, k) against `signs` (k, k) and `Vt * signs`
    # with `Vt` (n, k), so for m != n BOTH the reference and the port raise a
    # shape error (verified) — ASEBO's SVD block only ever runs square, i.e.
    # with subspace_dims == dim.  And such an X is ALWAYS rank deficient: its
    # columns sum to zero, so 1 is a left null vector, the last singular value
    # is 0, the last column of U is the constant vector, and its
    # `argmax(|U[:, -1]|)` is a float32-noise-level tie.  A zero singular value
    # cannot couple its left and right vectors either, so the sign of the right
    # null vector is free as well.  Both arbitrary choices differ between
    # numpy's and torch's SVD: of 25 generic dense X (measured), 16 make the two
    # pipelines differ by 1e-1..1.0 in UUT_ort although the port's ops are an
    # exact copy of the reference's.  `_asebo_subspace_input` therefore builds
    # X = U diag(s) V^T such that neither choice can reach
    # `UUT_ort = Vt[half:].T @ Vt[half:]` (the right null vector lives in the
    # first `half` rows, so the null direction never enters the [half:] block,
    # and V's last data row is a coordinate vector, whose outer product is
    # invariant under the arbitrary elementwise sign row) — verified
    # reproducible over 20 independent random draws of the construction (max
    # 1.5e-6, the gap-limited accuracy of the null vector).  The guard in (c)
    # shows this input still discriminates the pre-fix formulations.
    dense_x = _asebo_subspace_input(dim, half)
    probe = _drive(
        asebo,
        config,
        1,
        state_transform=lambda s: replace(
            s, grad_subspace=etl.ops.constant(etl.core.tensor(dense_x))
        ),
    )
    X = (dense_x - dense_x.mean(axis=0, dtype=F32)).astype(F32)
    reference = _torch_asebo_subspace_block(X, half)
    assert _max_abs_diff(probe.UUT_ort, reference) <= 1e-5

    # (c) discriminating-power guard: the two pre-fix formulations must differ
    # from the reference by far more than the tolerance asserted above.
    no_transpose = _torch_asebo_subspace_block(X, half, transpose=False)
    sign_vector = _torch_asebo_subspace_block(X, half, sign_matrix=False)
    assert _max_abs_diff(no_transpose, reference) > 1e-2
    assert _max_abs_diff(sign_vector, reference) > 1e-2

    # (d) horizon.  A `subspace_dims` (= dim) generation run on Sphere dim 10
    # still returns a finite best_fitness.  Beyond that horizon the TORCH
    # reference itself CRASHES: its alpha divides by the never-refreshed
    # all-zero `self.UUT`, so `torch.linalg.cholesky` raises LinAlgError at
    # generation `subspace_dims + 2` while numpy NaN-propagates and the port
    # fails one generation later — longer runs are deliberately not asserted.
    horizon = _drive(asebo, config, dim)
    assert np.isfinite(_array(horizon.best_fitness))


# --------------------------------------------------------------------------- #
# 2. ARS
# --------------------------------------------------------------------------- #
def test_ars_odd_pop_size_parity_with_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """ARS: the float-division elite count for ODD pop_size + even control."""
    dim, half = 10, 2
    center0 = np.full((dim,), 5.0, dtype=np.float32)
    # ARS samples `random.normal(subkey, (half, dim))` (half = pop_size // 2).
    _inject_fixed_noise(monkeypatch, _fixed_noise(2, (half, dim)))

    # ODD pop_size = 5: torch computes `max(1, int(pop_size / 2 * elite_ratio))`
    # = max(1, int(2.25)) = 2 == half, so EVERY mirrored pair is selected and
    # `fit_diff_noise` is an order-invariant sum — `argsort` tie order cannot
    # matter here.
    torch_odd = ARS(
        pop_size=5,
        center_init=torch.full((dim,), 5.0),
        elite_ratio=0.9,
        lr=0.05,
        sigma=0.5,
    )
    torch_odd.evaluate = lambda p: (p * p).sum(-1)
    torch_odd.step()
    assert torch_odd.elite_pop_size == half == 2

    cfg_odd = ars.make_ars(
        pop_size=5, center_init=center0, elite_ratio=0.9, lr=0.05, sigma=0.5
    )
    state_odd = _drive(ars, cfg_odd, 1)
    assert _max_abs_diff(state_odd.center, torch_odd.center) <= 1e-5
    # Discriminating power (throwaway experiment): with the OLD integer-division
    # rule `max(1, int((pop_size // 2) * elite_ratio))` = max(1, int(2 * 0.9))
    # = 1 the same module/run differs from the torch centre by |dcenter| =
    # 1.84e-2 (measured by re-running this module with that elite count
    # substituted, identical injected noise) — ~3 orders of magnitude above the
    # tolerance asserted above.

    # EVEN-pop_size CONTROL: pop_size = 4 -> `max(1, int(2 * 0.9))` = 1 elite
    # pair under BOTH the old and the new rule, so no regression can hide here.
    torch_even = ARS(
        pop_size=4,
        center_init=torch.full((dim,), 5.0),
        elite_ratio=0.9,
        lr=0.05,
        sigma=0.5,
    )
    torch_even.evaluate = lambda p: (p * p).sum(-1)
    torch_even.step()
    assert torch_even.elite_pop_size == 1

    cfg_even = ars.make_ars(
        pop_size=4, center_init=center0, elite_ratio=0.9, lr=0.05, sigma=0.5
    )
    state_even = _drive(ars, cfg_even, 1)
    assert _max_abs_diff(state_even.center, torch_even.center) <= 1e-5


# --------------------------------------------------------------------------- #
# 3. CMA-ES
# --------------------------------------------------------------------------- #
def test_cma_es_step_parity_with_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """CMA-ES: the rank-one covariance update (`torch.outer` vs scalar dot)."""
    dim = 10
    mean_init = np.linspace(-2.0, 2.0, dim).astype(np.float32)
    pop_size = 4 + int(np.floor(3 * np.log(dim)))  # 10 == torch's derived default
    # CMA-ES samples `random.normal(subkey, (pop_size, dim))`.
    _inject_fixed_noise(monkeypatch, _fixed_noise(11, (pop_size, dim)))

    torch_alg = CMAES(mean_init=torch.from_numpy(mean_init.copy()), sigma=2.0)
    assert torch_alg.pop_size == pop_size
    torch_alg.evaluate = lambda p: (p * p).sum(-1)
    # torch-2.12 environment workaround (verified necessary): `CMAES.step` ->
    # `_conditional_decomposition` calls `torch.cond(...)`, which raises
    # `UncapturedHigherOrderOpError` while capturing the `_no_decomposition`
    # branch.  `iteration % decomp_per_iter == 0` is always true for
    # `decomp_per_iter == 1` (asserted below), so `_decomposition` is always the
    # taken branch and calling it directly is EXACTLY equivalent.
    assert int(torch_alg.decomp_per_iter) == 1
    monkeypatch.setattr(
        CMAES,
        "_conditional_decomposition",
        lambda self, iteration, C: self._decomposition(C),
    )
    torch_alg.step()
    torch_alg.step()

    config = cma_es.make_cma_es(mean_init=mean_init, sigma=2.0)
    state = _drive(cma_es, config, 2)

    for name, etl_value, torch_value in (
        ("C", state.C, torch_alg.C),
        ("mean", state.mean, torch_alg.mean),
        ("sigma", state.sigma, torch_alg.sigma),
        ("p_c", state.p_c, torch_alg.p_c),
        ("p_sigma", state.p_sigma, torch_alg.p_sigma),
        ("C_invsqrt", state.C_invsqrt, torch_alg.C_invsqrt),
    ):
        assert _max_abs_diff(etl_value, torch_value) <= 1e-4, name

    # `B` and `D` are NOT element-wise comparable (measured |dB| = 1.60,
    # |dD| = 1.60 for this config).  `_decomposition` eigendecomposes C with
    # `eigh`, and at iteration 2 the covariance update is a rank-(mu + 1)
    # perturbation of a multiple of the identity, so several eigenvalues of C
    # coincide to float32 precision (min measured gap 0.0 here; 3.6e-3 even for
    # pop_size = 20): within such an eigenspace the eigenBASIS is not unique.
    # numpy's and torch's `eigh` additionally disagree on the eigenvector SIGN
    # convention for individual columns.  What IS unique — and is asserted
    # instead — is exactly what the sampling step consumes:
    #   * `D @ B == C^{1/2}`, the symmetric PSD square root of C, and
    #   * `B @ D == diag(sqrt(eigenvalues of C))`,
    # both basis- and sign-independent (the state stores B == B_vecs.T, i.e. the
    # same layout as the torch reference).
    assert _max_abs_diff(
        _array(state.D) @ _array(state.B),
        _array(torch_alg.D) @ _array(torch_alg.B),
    ) <= 1e-4
    assert _max_abs_diff(
        _array(state.B) @ _array(state.D),
        _array(torch_alg.B) @ _array(torch_alg.D),
    ) <= 1e-4
    B = _array(state.B)
    assert np.max(np.abs(B @ B.T - np.eye(dim))) <= 1e-4

    # Discriminating power (throwaway experiment): restoring the pre-fix scalar
    # dot form (`pc_norm_sq = etl.sum(p_c * p_c)`, a 0-d value broadcast
    # additively into every element of C instead of the true outer product) and
    # re-running with the SAME injected noise changes C by |dC| = 0.1197 after 2
    # generations — ~6 orders of magnitude above the 5.96e-8 float32 rounding
    # difference the true `torch.outer` form shows against the reference.

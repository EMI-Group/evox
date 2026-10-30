"""Functional ETL port of the torch evox ASEBO algorithm (step protocol).

Reference (read-only): ``src/evox/algorithms/so/es_variants/asebo.py``
(Adaptive ES-Active Subspaces for Blackbox Optimization).  EvoX Compiler has no eager
mode, so ``init``/``step`` are plain functions traced via ``evoxc.build``/
``evoxc.run``; ``step`` owns the WHOLE generation — the torch ``step`` body with
its internal ``self.evaluate`` call fused in via the workflow-injected
``evaluate`` closure.  ``lr_decay``/``lr_limit`` are kept for API parity but
unused (as in torch).
"""

from dataclasses import dataclass, replace
from typing import Callable, Literal

import evoxc
import evoxc.numpy as enp
import evoxc.random as random
import numpy as np
from evoxc.core import SymbolicTensor

from evox_etl.algorithms._config_utils import (
    ArrayLike,
    bake_float32_constant,
    require_choice,
    require_gt,
    to_float_tuple,
)

from .adam_step import adam_single_tensor

F32 = np.dtype("float32")


@dataclass(frozen=True)
class ASEBOConfig:
    """Hyperparameters of the ASEBO algorithm (mirrors the torch ``__init__``).

    Dumb frozen config storing plain static leaves: `center_init` arrives as a
    flat float32 tuple, normalized by `make_asebo`, which also derives
    `subspace_dims` from it eagerly when not given.
    """

    pop_size: int
    center_init: tuple[float, ...]
    optimizer: Literal["adam"] | None = None
    lr: float = 0.05
    lr_decay: float = 1.0
    lr_limit: float = 0.001
    sigma: float = 0.03
    sigma_decay: float = 1.0
    sigma_limit: float = 0.01
    subspace_dims: int | None = None


def make_asebo(
    pop_size: int,
    center_init: ArrayLike,
    optimizer: Literal["adam"] | None = None,
    lr: float = 0.05,
    lr_decay: float = 1.0,
    lr_limit: float = 0.001,
    sigma: float = 0.03,
    sigma_decay: float = 1.0,
    sigma_limit: float = 0.01,
    subspace_dims: int | None = None,
) -> ASEBOConfig:
    """Build an `ASEBOConfig`, normalizing `center_init` and validating fields.

    `subspace_dims` defaults to ``len(center_init)`` when not given (derived
    eagerly here so ``cfg.subspace_dims`` is always concrete).  `lr_decay` and
    `lr_limit` are kept for API parity but unused (as in torch).
    """
    require_gt("pop_size", pop_size, 1)
    require_choice("optimizer", optimizer, (None, "adam"))
    center_init = to_float_tuple(center_init, dtype=np.float32)
    if subspace_dims is None:
        subspace_dims = len(center_init)
    return ASEBOConfig(
        pop_size=pop_size,
        center_init=center_init,
        optimizer=optimizer,
        lr=lr,
        lr_decay=lr_decay,
        lr_limit=lr_limit,
        sigma=sigma,
        sigma_decay=sigma_decay,
        sigma_limit=sigma_limit,
        subspace_dims=subspace_dims,
    )


@dataclass(frozen=True)
class ASEBOState:
    """Mutable state of ASEBO; all leaves are evoxc tensors."""

    center: SymbolicTensor
    grad_subspace: SymbolicTensor
    UUT: SymbolicTensor
    UUT_ort: SymbolicTensor
    sigma: SymbolicTensor
    alpha: SymbolicTensor
    gen_counter: SymbolicTensor
    z: SymbolicTensor
    exp_avg: SymbolicTensor
    exp_avg_sq: SymbolicTensor
    best_fitness: SymbolicTensor
    key: SymbolicTensor


def init(config: ASEBOConfig, key: SymbolicTensor) -> ASEBOState:
    """Create the initial state; ``center`` is baked as a graph constant."""
    dim = len(config.center_init)
    center = bake_float32_constant(config.center_init)
    grad_subspace = enp.zeros((config.subspace_dims, dim), dtype=F32)
    UUT = enp.zeros((dim, dim), dtype=F32)
    UUT_ort = enp.zeros((dim, dim), dtype=F32)
    sigma = enp.full((), config.sigma, dtype=F32)
    alpha = enp.full((), 0.1, dtype=F32)
    gen_counter = enp.full((), 0.0, dtype=F32)
    z = enp.zeros((config.pop_size, dim), dtype=F32)
    exp_avg = enp.zeros((dim,), dtype=F32)
    exp_avg_sq = enp.zeros((dim,), dtype=F32)
    best_fitness = enp.full((), np.inf, dtype=F32)
    return ASEBOState(
        center,
        grad_subspace,
        UUT,
        UUT_ort,
        sigma,
        alpha,
        gen_counter,
        z,
        exp_avg,
        exp_avg_sq,
        best_fitness,
        key,
    )


def step(
    config: ASEBOConfig,
    state: ASEBOState,
    evaluate: Callable[[SymbolicTensor], SymbolicTensor],
) -> ASEBOState:
    """Run ONE full generation: rebuild the active subspace from the gradient
    history, sample the mirrored population from the covariance it induces,
    evaluate it, and update center, alpha and sigma from the mirrored fitness
    pairs.

    The subspace basis is the leading ``subspace_dims`` rows of the right
    singular vectors ``Vh`` of the CENTRED gradient buffer; directions whose
    singular value is numerically zero are dropped (the centred buffer has rank
    at most ``subspace_dims - 1``), ``UUT = U^T U`` is the projector onto the
    subspace and ``UUT_ort = I - UUT`` its orthogonal complement.  Both are
    zeroed — and ``alpha`` is pinned to 1.0 — until more than
    ``subspace_dims`` gradients have been buffered (determined ONCE per
    generation from the pre-increment counter).  The freshly computed ``UUT``/
    ``UUT_ort`` are stored in the returned state and used for this generation's
    alpha.

    ``evaluate`` is the workflow-injected traced closure (opaque; minimization
    semantics).
    """
    dim = len(config.center_init)
    half = config.pop_size // 2
    sub = config.subspace_dims

    key, subkey = random.split(state.key)

    # Gate computed ONCE, from the PRE-increment counter, and reused for both
    # the UUT/UUT_ort mask and the alpha mask (mirrors the torch reference).
    subspace_ready = evoxc.greater(state.gen_counter, float(sub))

    # Active subspace via SVD of the centred gradient history.
    X = state.grad_subspace - evoxc.mean(state.grad_subspace, axes=0)
    # evoxc.svd follows the numpy / `torch.linalg.svd` convention and returns a
    # reduced (U, S, Vh) triple: the ROWS of Vh are the right singular vectors,
    # i.e. the principal directions of the past gradients.
    _, S, Vh = evoxc.svd(X)
    U = Vh[:sub]  # (k, dim), k = min(sub, dim) — no transpose / sign juggling
    # Drop directions without variance: the centred buffer has rank at most
    # `subspace_dims - 1`, so the trailing singular value(s) are ~0.  `tol` is
    # the numpy/torch SVD rank tolerance; only the eps is a Python float.
    tol = float(np.finfo(np.float32).eps) * max(sub, dim) * S[0]
    informative = evoxc.cast(evoxc.greater(S[:sub], tol), F32)  # (k,)

    # UUT = U^T diag(informative) U, broadcasting (dim, k) * (k,) over the last axis.
    UUT = evoxc.dot(evoxc.transpose(U, (1, 0)) * informative, U)  # (dim, dim)
    UUT_ort = evoxc.eye(dim) - UUT
    zero = enp.zeros((dim, dim), dtype=F32)
    UUT = evoxc.select(subspace_ready, UUT, zero)
    UUT_ort = evoxc.select(subspace_ready, UUT_ort, zero)
    # Number of genuinely informative directions, floored at 1 (scalar).
    effective_dims = evoxc.maximum(evoxc.sum(informative, axes=0), 1.0)

    # Perturbations are sigma * N(0, cov), as in evosax (which the reference follows).
    cov = (
        (state.alpha / dim) * evoxc.eye(dim)
        + ((1.0 - state.alpha) / effective_dims) * UUT
        + 1e-6 * evoxc.eye(dim)
    )
    chol = evoxc.cholesky(cov)
    noise = random.normal(subkey, (half, dim), mean=0.0, std=1.0, dtype=F32)
    z_plus = state.sigma * evoxc.dot(noise, evoxc.transpose(chol, (1, 0)))
    z = evoxc.concatenate([z_plus, -1.0 * z_plus], axis=0)
    population = state.center + z
    gen_counter = state.gen_counter + 1.0
    state = replace(
        state, z=z, UUT=UUT, UUT_ort=UUT_ort, gen_counter=gen_counter, key=key
    )

    fitness = evaluate(population)

    noise_1 = z[:half] / state.sigma
    fit_diff_noise = evoxc.dot(
        enp.expand_dims(fitness[:half] - fitness[half:], axis=0), noise_1
    )[0]
    theta_grad = 0.5 * fit_diff_noise
    # alpha is computed from THIS generation's freshly computed UUT/UUT_ort
    # (the pre-fix port divided by the never-written `state.UUT` buffer, which
    # made alpha NaN — and the next Cholesky factorisation fail — as soon as the
    # subspace activated).
    alpha = evoxc.norm(
        evoxc.dot(enp.expand_dims(theta_grad, axis=0), UUT_ort)[0]
    ) / (evoxc.norm(evoxc.dot(enp.expand_dims(theta_grad, axis=0), UUT)[0]) + 1e-8)
    alpha = evoxc.select(subspace_ready, evoxc.clamp(alpha, 0.0, 1.0), 1.0)
    grad_subspace = evoxc.concatenate(
        [state.grad_subspace, enp.expand_dims(theta_grad, axis=0)], axis=0
    )[1:, :]
    theta_grad = theta_grad / (evoxc.norm(theta_grad) / dim + 1e-8)
    if config.optimizer == "adam":
        center, exp_avg, exp_avg_sq = adam_single_tensor(
            state.center,
            theta_grad,
            state.exp_avg,
            state.exp_avg_sq,
            beta1=0.9,
            beta2=0.999,
            lr=config.lr,
        )
    else:
        center = state.center - config.lr * theta_grad
        exp_avg, exp_avg_sq = state.exp_avg, state.exp_avg_sq
    sigma = evoxc.maximum(state.sigma * config.sigma_decay, config.sigma_limit)
    best_fitness = evoxc.minimum(state.best_fitness, evoxc.min(fitness))
    return replace(
        state,
        center=center,
        sigma=sigma,
        alpha=alpha,
        grad_subspace=grad_subspace,
        exp_avg=exp_avg,
        exp_avg_sq=exp_avg_sq,
        best_fitness=best_fitness,
    )


def record_step(
    config: ASEBOConfig,
    state: ASEBOState,
    candidate: SymbolicTensor | None,
    fitness: SymbolicTensor | None,
) -> dict[str, SymbolicTensor]:
    """Host-side auxiliary-history hook (mirrors torch ``ASEBO.record_step``).

    Runs OUTSIDE any traced graph on the CONCRETE post-step state; ``candidate``
    and ``fitness`` are accepted for signature parity with the other algorithms
    and ignored (torch ignores them too).  Returns the current search
    distribution as ``{"center": state.center, "sigma": state.sigma, "alpha":
    state.alpha}``, which the workflow forwards to the monitor's
    ``record_auxiliary`` hook.
    """
    return {
        "center": state.center,
        "sigma": state.sigma,
        "alpha": state.alpha,
    }

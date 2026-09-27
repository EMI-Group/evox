"""Pure-etl tests for the EvalMonitor shape policy (torch EvalMonitor parity).

The monitor stores the FULL evaluated batch as-is: the candidate/fitness
leading dim may differ from ``pop_size`` (CoDE evaluates ``3 * pop_size``
trials per generation, CSO ``pop_size // 2`` after its first generation —
torch ``EvalMonitor.post_ask``/``pre_tell`` store the raw batch the same way,
with no slicing to ``pop_size``). These tests drive the module-level
``init``/``monitor_update`` functions directly through ``etl.build``/``etl.run``
with several batch leading dims, pinning the monitor semantics independently
of ``StdWorkflow``.

No torch imports — numpy backend only.
"""

from __future__ import annotations

import dataclasses

import etl
import etl.core
import numpy as np
import pytest

import evox_etl.workflows.eval_monitor as em

_KEY_SPEC = etl.core.TensorSpec((), np.int64)
_CPU = etl.core.Device("cpu")


def _run_init(cfg: em.EvalMonitorConfig):
    """Run the monitor init graph (mirrors StdWorkflow's monitor-init exe)."""
    def body(key):
        return em.init(cfg, key)

    exe = etl.build(body, _KEY_SPEC, backend="numpy", device=_CPU)
    return etl.run(exe, etl.core.tensor(np.asarray(0, dtype=np.int64)))


def _spec_tree(pytree):
    return etl.tree_map(
        lambda t: etl.core.TensorSpec(tuple(t.shape), t.dtype), pytree
    )


def _update(cfg: em.EvalMonitorConfig, state, cand: np.ndarray, fit: np.ndarray):
    """Run one monitor_update graph on concrete tensors (cfg closed over,
    mirroring the workflow — the config carries host-side list fields)."""
    def body(st, c, f):
        return em.monitor_update(cfg, st, c, f)

    specs = (
        _spec_tree(state),
        etl.core.TensorSpec(cand.shape, np.float32),
        etl.core.TensorSpec(fit.shape, np.float32),
    )
    exe = etl.build(body, *specs, backend="numpy", device=_CPU)
    return etl.run(exe, state, etl.core.tensor(cand), etl.core.tensor(fit))


def _to_np(t):
    return np.asarray(t.to(_CPU).numpy())


def _so_cfg(pop_size=10, dim=3, topk=2, **kwargs):
    return em.EvalMonitorConfig(
        pop_size=pop_size, dim=dim, topk=topk, multi_obj=False, **kwargs
    )


# --------------------------------------------------------------------- init


def test_init_allocates_pop_size_placeholder_buffers():
    cfg = _so_cfg(pop_size=10, dim=3, topk=2)
    state = _run_init(cfg)
    assert isinstance(state, em.EvalMonitorState)
    assert _to_np(state.latest_solution).shape == (10, 3)
    assert _to_np(state.latest_fitness).shape == (10,)
    assert _to_np(state.topk_solutions).shape == (2, 3)
    # +inf elite init makes the running top-k monotone (gen-0 always wins)
    assert np.all(np.isinf(_to_np(state.topk_fitness)))


def test_init_requires_pop_size_and_dim():
    with pytest.raises(ValueError):
        _run_init(em.EvalMonitorConfig(multi_obj=False))


def test_mo_init_requires_n_obj():
    with pytest.raises(ValueError):
        _run_init(
            em.EvalMonitorConfig(multi_obj=True, pop_size=8, dim=4, n_obj=None)
        )


# ------------------------------------------------- SO: any leading dimension


def test_so_update_stores_full_wide_batch():
    """CoDE case: a (3*pop_size, dim) batch is stored whole (torch parity)."""
    pop, dim = 10, 3
    cfg = _so_cfg(pop_size=pop, dim=dim, topk=2)
    state = _run_init(cfg)
    cand = np.arange(30 * dim, dtype=np.float32).reshape(30, dim)
    fit = np.linspace(5.0, 1.0, 30, dtype=np.float32)  # strictly decreasing
    out = _update(cfg, state, cand, fit)
    assert _to_np(out.latest_solution).shape == (30, dim)
    assert _to_np(out.latest_fitness).shape == (30,)
    # the elite pools the ENTIRE evaluated batch, not a pop_size slice
    assert _to_np(out.topk_fitness).shape == (2,)
    assert _to_np(out.topk_fitness)[0] == pytest.approx(fit.min())
    assert np.allclose(_to_np(out.topk_solutions)[0], cand[np.argmin(fit)])


def test_so_update_stores_narrow_batch():
    """CSO case: a (pop_size // 2, dim) batch is stored whole (torch parity)."""
    cfg = _so_cfg(pop_size=10, dim=3, topk=2)
    state = _run_init(cfg)
    cand = np.ones((5, 3), dtype=np.float32)
    fit = np.full(5, 2.0, dtype=np.float32)
    out = _update(cfg, state, cand, fit)
    assert _to_np(out.latest_solution).shape == (5, 3)
    assert _to_np(out.latest_fitness).shape == (5,)
    assert _to_np(out.topk_fitness).shape == (2,)


def test_so_elite_pools_across_drifting_batch_sizes():
    """The running elite survives leading-dim changes between generations."""
    cfg = _so_cfg(pop_size=10, dim=2, topk=2)
    state = _run_init(cfg)
    # generation 1: 30 candidates, best value 0.5 at a known row
    cand1 = np.zeros((30, 2), dtype=np.float32)
    fit1 = np.full(30, 4.0, dtype=np.float32)
    fit1[7] = 0.5
    state = _update(cfg, state, cand1, fit1)
    # generation 2: only 5 candidates, none beats 0.5
    cand2 = np.zeros((5, 2), dtype=np.float32)
    fit2 = np.full(5, 1.5, dtype=np.float32)
    fit2[2] = 0.75
    state = _update(cfg, state, cand2, fit2)
    topk = np.sort(_to_np(state.topk_fitness))
    assert topk[0] == pytest.approx(0.5)
    assert topk[1] == pytest.approx(0.75)


# ------------------------------------------------------------------------ MO


def test_mo_update_stores_full_batch():
    cfg = em.EvalMonitorConfig(
        pop_size=8, dim=4, n_obj=3, multi_obj=True, topk=1
    )
    state = _run_init(cfg)
    assert isinstance(state, em.MOEvalMonitorState)
    cand = np.zeros((24, 4), dtype=np.float32)  # 3 * pop_size (NSGA-style merge)
    fit = np.ones((24, 3), dtype=np.float32)
    out = _update(cfg, state, cand, fit)
    assert isinstance(out, em.MOEvalMonitorState)
    assert _to_np(out.latest_solution).shape == (24, 4)
    assert _to_np(out.latest_fitness).shape == (24, 3)


# --------------------------------------------------------- wrapper accessors


def test_wrapper_accessors_after_wide_batch():
    cfg = _so_cfg(pop_size=6, dim=2, topk=1, opt_direction=(1,))
    state = _run_init(cfg)
    cand = np.arange(18 * 2, dtype=np.float32).reshape(18, 2)
    fit = np.linspace(3.0, 0.5, 18, dtype=np.float32)
    state = _update(cfg, state, cand, fit)
    mon = em.EvalMonitor(config=cfg, state=state)
    assert isinstance(mon.get_best_fitness(), float)
    assert mon.get_best_fitness() == pytest.approx(fit.min())
    assert mon.get_latest_fitness().shape == (18,)
    assert mon.get_topk_fitness().shape == (1,)
    assert mon.get_best_solution().shape == (2,)
    assert dataclasses.is_dataclass(mon.config)


def test_wrapper_un_negates_max_direction():
    cfg = _so_cfg(pop_size=6, dim=2, topk=1, opt_direction=(-1,))
    state = _run_init(cfg)
    cand = np.zeros((6, 2), dtype=np.float32)
    fit = np.array([-1.0, -2.0, -3.0, -0.5, -4.0, -2.5], dtype=np.float32)
    state = _update(cfg, state, cand, fit)
    mon = em.EvalMonitor(config=cfg, state=state)
    # stored (minimized) best is -4.0; the accessor un-negates to +4.0
    assert mon.get_best_fitness() == pytest.approx(4.0)
    assert mon.get_latest_fitness()[2] == pytest.approx(3.0)


def test_wrapper_so_accessor_raises_under_mo():
    cfg = em.EvalMonitorConfig(pop_size=8, dim=4, n_obj=2, multi_obj=True)
    state = _run_init(cfg)
    mon = em.EvalMonitor(config=cfg, state=state)
    with pytest.raises(ValueError):
        mon.get_best_fitness()
    with pytest.raises(ValueError):
        mon.get_topk_solutions()

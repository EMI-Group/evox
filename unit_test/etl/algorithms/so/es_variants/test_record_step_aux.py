"""End-to-end: an ES-variant's module-level ``record_step`` aux hook reaches
``EvalMonitor.aux_history`` through ``StdWorkflow``.

Aux-channel path (``src/evox_etl/core/workflow.py`` ``_record_history``):
``StdWorkflow`` resolves the algorithm's plain functions via
``type(config).__module__``; when that module defines a module-level host-side
``record_step(config, state, candidate, fitness)`` hook AND the monitor module
defines ``record_auxiliary``, the workflow calls the hook after each generation
on the concrete POST-step state and forwards its dict to the monitor. The
monitor copies each value to CPU numpy and appends it under the dict's key (one
entry per generation), gated on ``EvalMonitorConfig.full_pop_history``.

Every module under ``.../so/es_variants/`` defines ``record_step``. This test
drives ``cma_es`` (``{"mean": (1, dim), "sigma": scalar}``) and ``des``
(``{"center": (dim,), "sigma": (dim,)}``) through ``StdWorkflow`` + the paired
``evox_etl.problems.numerical.basic.Sphere``.

NOTE: these ES states expose neither ``population``/``pop`` (so the workflow
cannot infer the dimension from the state) nor a ``dim`` config field, so
``dim`` is passed to ``EvalMonitorConfig`` explicitly — otherwise monitor init
raises ``ValueError`` (pop_size, by contrast, is a config field the workflow
reads; cma_es' ``pop_size=None`` default is NOT derived host-side).

Numpy backend only; no torch imports.
"""

from __future__ import annotations

import numpy as np

from evox_etl.algorithms.so.es_variants import cma_es, des
from evox_etl.problems.numerical.basic import Sphere
from evox_etl.workflows import EvalMonitorConfig, StdWorkflow

DIM, POP, GENS = 4, 8, 6


def _run(algo_cfg):
    """Drive ``algo_cfg`` through StdWorkflow + Sphere with an aux-recording monitor."""
    wf = StdWorkflow(
        algo_cfg,
        Sphere(),
        monitor=EvalMonitorConfig(full_pop_history=True, dim=DIM, pop_size=POP),
        opt_direction="min",
        num_generations=GENS,
    )
    wf.run()
    return wf


def _cma_es_workflow():
    cfg = cma_es.make_cma_es(
        mean_init=np.full((DIM,), 2.0, np.float32), sigma=1.0, pop_size=POP
    )
    return _run(cfg)


def test_cma_es_aux_keys_match_record_step_output():
    """``aux_history`` keys are EXACTLY ``cma_es.record_step``'s, one entry/gen."""
    wf = _cma_es_workflow()
    aux = wf.monitor.aux_history
    assert aux is wf.monitor_config.aux_history
    final = wf._state.algorithm_state
    assert set(aux) == set(cma_es.record_step(wf.algorithm, final, None, None))
    assert set(aux) == {"mean", "sigma"}
    for key in ("mean", "sigma"):
        assert len(aux[key]) == GENS
        assert all(isinstance(entry, np.ndarray) for entry in aux[key])


def test_cma_es_aux_entry_shapes_and_finiteness():
    """``mean`` entries are (1, dim), ``sigma`` entries are finite scalars."""
    aux = _cma_es_workflow().monitor.aux_history
    assert all(entry.shape == (1, DIM) for entry in aux["mean"])
    assert all(np.all(np.isfinite(entry)) for entry in aux["mean"])
    assert all(entry.shape == () for entry in aux["sigma"])
    assert all(np.isfinite(entry) for entry in aux["sigma"])


def test_cma_es_aux_last_entry_matches_final_algorithm_state():
    """The last recorded entry IS the post-run algorithm state (mean/sigma)."""
    wf = _cma_es_workflow()
    aux = wf.monitor.aux_history
    final = wf._state.algorithm_state  # post-run state (StdWorkflow._run_variant)
    assert np.allclose(aux["mean"][-1], np.asarray(final.mean.numpy()))
    assert np.allclose(aux["sigma"][-1], np.asarray(final.sigma.numpy()))


def test_cma_es_recorded_sigma_is_evolving_not_static_config_value():
    """The recorded ``sigma`` evolves — it is NOT the static config value."""
    wf = _cma_es_workflow()
    sigma_series = np.array([np.asarray(s) for s in wf.monitor.aux_history["sigma"]])
    assert not np.allclose(sigma_series, sigma_series[0])  # varies across generations
    assert not np.allclose(sigma_series, wf.algorithm.sigma)  # not the pinned config


def test_des_aux_center_and_sigma_match_final_state():
    """Breadth: ``des``' ``{"center","sigma"}`` hook lands in ``aux_history`` too."""
    cfg = des.make_des(
        pop_size=POP, center_init=np.full((DIM,), 2.0, np.float32), sigma_init=1.0
    )
    wf = _run(cfg)
    aux = wf.monitor.aux_history
    assert set(aux) == {"center", "sigma"}
    for key in ("center", "sigma"):
        assert len(aux[key]) == GENS
        assert all(isinstance(entry, np.ndarray) for entry in aux[key])
        assert all(entry.shape == (DIM,) for entry in aux[key])
    final = wf._state.algorithm_state
    assert np.allclose(aux["center"][-1], np.asarray(final.center.numpy()))
    assert np.allclose(aux["sigma"][-1], np.asarray(final.sigma.numpy()))

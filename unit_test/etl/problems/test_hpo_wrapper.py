"""Pure-etl smoke tests for the host-side HPO wrapper (NO torch imports).

Builds small inner ``StdWorkflow``s (PSO on Sphere with an ``EvalMonitorConfig``),
wraps them in :class:`HPOProblemWrapper`, and checks that ``evaluate`` really runs
the inner workflow with the substituted hyperparameters (compared against a
manually built workflow with the same seed/backend), that the repeat/aggregation
policy behaves as documented, and that ``random_search`` returns the best sampled
candidate.  Everything runs on the numpy backend.
"""

import numpy as np
import pytest

from evox_etl import EvalMonitorConfig, StdWorkflow
from evox_etl.algorithms import NSGA2, PSO
from evox_etl.problems.hpo_wrapper import (
    HPOFitnessMonitor,
    HPOFitnessMonitorConfig,
    HPOMonitor,
    HPOProblemConfig,
    HPOProblemWrapper,
    HPSlot,
    random_search,
)
from evox_etl.problems.numerical import DTLZ2, Sphere

DIM = 3
LB = (-5.12,) * DIM
UB = (5.12,) * DIM
POP_SIZE = 20
GENERATIONS = 5

DEFAULT_SLOTS = (HPSlot("algorithm", "w"), HPSlot("algorithm", "phi_p"))


def build_wrapper(
    hyperparameters=DEFAULT_SLOTS,
    num_generations=GENERATIONS,
    num_repeats=1,
    seed=0,
    fit_aggregation="mean",
):
    """Build a PSO-on-Sphere HPO wrapper (two tuned PSO weights by default)."""
    return HPOProblemWrapper(
        HPOProblemConfig(
            algorithm=PSO(pop_size=POP_SIZE, lb=LB, ub=UB),
            problem=Sphere(),
            hyperparameters=hyperparameters,
            num_generations=num_generations,
            monitor=HPOFitnessMonitorConfig(inner_monitor=EvalMonitorConfig(), fit_aggregation=fit_aggregation),
            num_repeats=num_repeats,
            seed=seed,
        )
    )


def run_inner(**algorithm_fields):
    """Run the equivalent inner workflow manually (same seed/backend)."""
    fields = {"pop_size": POP_SIZE, "lb": LB, "ub": UB, **algorithm_fields}
    workflow = StdWorkflow(
        algorithm=PSO(**fields),
        problem=Sphere(),
        monitor=EvalMonitorConfig(),
        opt_direction="min",
        backend="numpy",
    )
    workflow.run(GENERATIONS, seed=0)
    return workflow.monitor.get_best_fitness()


# --- accessors ------------------------------------------------------------


def test_get_params_keys_and_init_params():
    wrapper = build_wrapper()
    assert wrapper.num_hyperparameters == 2
    assert wrapper.get_params_keys() == ["algorithm.w", "algorithm.phi_p"]
    assert wrapper.get_init_params() == {"algorithm.w": 0.6, "algorithm.phi_p": 2.5}


# --- evaluate: end-to-end inner run ---------------------------------------


def test_evaluate_vector_matches_manual_inner_run():
    fitness = build_wrapper().evaluate([0.4, 1.0])
    assert isinstance(fitness, float)
    assert np.isfinite(fitness)
    assert fitness >= 0.0  # Sphere is non-negative
    # exact parity: same seed, same backend, same substituted hyperparameters
    assert fitness == run_inner(w=0.4, phi_p=1.0)


def test_evaluate_defaults_keep_inner_config_values():
    wrapper = build_wrapper()
    defaults = wrapper.evaluate(None)
    assert wrapper.evaluate({}) == defaults  # torch's evaluate({}) semantics
    # omitted mapping keys fall back to the inner config's current value
    assert wrapper.evaluate({"algorithm.w": 0.6}) == wrapper.evaluate({"algorithm.w": 0.6, "algorithm.phi_p": 2.5})
    # ... and passing the defaults explicitly is the same run
    assert wrapper.evaluate({"algorithm.w": 0.6, "algorithm.phi_p": 2.5}) == defaults


def test_slot_transform_is_applied():
    wrapper = build_wrapper(hyperparameters=(HPSlot("algorithm", "pop_size", transform=lambda value: int(round(value))),))
    assert wrapper.get_init_params() == {"algorithm.pop_size": POP_SIZE}
    # raw 12.4 would be an invalid shape without the int transform
    fitness = wrapper.evaluate([12.4])
    assert fitness == run_inner(pop_size=12)


def test_repeats_use_consecutive_seeds_and_aggregate():
    vector = [0.5, 1.5]
    repeat0 = build_wrapper(num_repeats=1, seed=7)
    repeat1 = build_wrapper(num_repeats=1, seed=8)
    expected_mean = 0.5 * (repeat0.evaluate(vector) + repeat1.evaluate(vector))

    mean_wrapper = build_wrapper(num_repeats=2, seed=7, fit_aggregation="mean")
    assert mean_wrapper.evaluate(vector) == pytest.approx(expected_mean, rel=1e-12)

    min_wrapper = build_wrapper(num_repeats=2, seed=7, fit_aggregation="min")
    assert min_wrapper.evaluate(vector) == pytest.approx(min(repeat0.evaluate(vector), repeat1.evaluate(vector)), rel=1e-12)


# --- multi-objective inner problems ---------------------------------------


def build_mo_wrapper(multi_obj_metric):
    """An NSGA2-on-DTLZ2 HPO wrapper (MO inner problem, one tuned pop_size slot)."""
    return HPOProblemWrapper(
        HPOProblemConfig(
            algorithm=NSGA2(pop_size=10, n_objs=2, lb=(0.0,) * 4, ub=(1.0,) * 4),
            problem=DTLZ2(d=4, m=2),
            hyperparameters=(HPSlot("algorithm", "pop_size", transform=lambda value: int(round(value))),),
            num_generations=3,
            monitor=HPOFitnessMonitorConfig(
                inner_monitor=EvalMonitorConfig(full_fit_history=True),
                multi_obj_metric=multi_obj_metric,
            ),
            opt_direction=["min", "min"],
            seed=0,
        )
    )


@pytest.mark.filterwarnings("ignore:divide by zero")
def test_multi_objective_inner_problem_scores_the_pareto_front():
    metric = lambda pf: np.linalg.norm(np.asarray(pf), axis=1)  # noqa: E731
    fitness = build_mo_wrapper(metric).evaluate([10.0])
    assert np.isfinite(fitness)
    # the DTLZ2 Pareto front is the unit sphere, so the smallest norm is ~1
    assert 0.9 < fitness < 1.5

    # without a metric the SO accessor would be meaningless -> loud ValueError
    with pytest.raises(ValueError, match="single best solution"):
        build_mo_wrapper(None).evaluate([10.0])


# --- monitor edge cases ---------------------------------------------------


def test_monitor_base_and_missing_inner_monitor_raise():
    with pytest.raises(NotImplementedError, match="tell_fitness"):
        HPOMonitor().tell_fitness(None)
    with pytest.raises(ValueError, match="requires the inner workflow"):
        HPOFitnessMonitor().tell_fitness(object())
    with pytest.raises(ValueError, match="must be None or callable"):
        HPOFitnessMonitor(multi_obj_metric=3)
    with pytest.raises(ValueError, match="num_repeats"):
        HPOMonitor(num_repeats=0)
    with pytest.raises(ValueError, match="fit_aggregation"):
        HPOMonitor(fit_aggregation="median")
    with pytest.raises(TypeError, match="HPOFitnessMonitorConfig"):
        HPOProblemWrapper(
            HPOProblemConfig(
                algorithm=PSO(pop_size=POP_SIZE, lb=LB, ub=UB),
                problem=Sphere(),
                monitor="not-a-config",
            )
        )


# --- error paths ----------------------------------------------------------


def test_unknown_key_and_wrong_vector_length_raise():
    wrapper = build_wrapper()
    with pytest.raises(ValueError, match="Unknown hyperparameter"):
        wrapper.evaluate({"algorithm.bogus": 1.0})
    with pytest.raises(ValueError, match="Expected 2 hyperparameter"):
        wrapper.evaluate([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="must be finite"):
        wrapper.evaluate([0.5, np.nan])


def test_invalid_slots_rejected():
    base = dict(algorithm=PSO(pop_size=POP_SIZE, lb=LB, ub=UB), problem=Sphere(), num_generations=2)
    with pytest.raises(ValueError, match="not a field of the inner algorithm config"):
        HPOProblemWrapper(HPOProblemConfig(**base, hyperparameters=(HPSlot("algorithm", "bogus"),)))
    with pytest.raises(ValueError, match="config_kind"):
        HPOProblemWrapper(HPOProblemConfig(**base, hyperparameters=(HPSlot("nope", "w"),)))
    with pytest.raises(ValueError, match="Duplicate hyperparameter slot"):
        HPOProblemWrapper(HPOProblemConfig(**base, hyperparameters=(HPSlot("algorithm", "w"), HPSlot("algorithm", "w"))))
    with pytest.raises(ValueError, match="num_generations"):
        HPOProblemWrapper(HPOProblemConfig(**{**base, "num_generations": 0}))


# --- random search driver -------------------------------------------------


def test_random_search_returns_best_candidate():
    wrapper = build_wrapper()
    best, fitness = random_search(wrapper, (0.0, 3.0), n_trials=3, seed=1)
    assert best.shape == (2,)
    assert np.all(best >= 0.0) and np.all(best <= 3.0)
    assert np.isfinite(fitness)
    # the reported fitness belongs to the reported candidate (deterministic)
    assert wrapper.evaluate(best) == fitness

    # per-slot bounds
    per_slot_best, per_slot_fitness = random_search(wrapper, [(0.0, 1.0), (1.0, 2.0)], n_trials=2, seed=3)
    assert 0.0 <= per_slot_best[0] <= 1.0
    assert 1.0 <= per_slot_best[1] <= 2.0
    assert wrapper.evaluate(per_slot_best) == per_slot_fitness

    with pytest.raises(ValueError, match="n_trials"):
        random_search(wrapper, (0.0, 1.0), n_trials=0)
    with pytest.raises(ValueError, match="bounds"):
        random_search(wrapper, [(0.0, 1.0)], n_trials=1)

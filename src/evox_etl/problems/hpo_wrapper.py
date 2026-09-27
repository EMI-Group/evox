"""Host-side functional hyperparameter optimization (HPO) for evox_etl.

Port of the torch ``src/evox/problems/hpo_wrapper.py`` (read-only reference) onto
the ETL functional design.  The torch original wraps a whole ``Workflow`` as an
``evox.core.Problem`` whose search space is the inner workflow's ``Parameter``s and
runs the inner workflow INSIDE the outer workflow's traced ``evaluate``
(``torch.func.stack_module_state`` + ``vmap`` + a ``torch.library`` custom-op loop,
bookkept in the module-global ``HPOData`` registry).

## Known Issues (why this is a host-side redesign rather than a port)

1. **A nested compiled workflow can never run inside an outer trace.**  ETL has NO
   eager mode (``DESIGN.md`` §2.1: every ``etl.numpy``/``etl.ops``/``etl.random``
   call needs an active trace) and ``etl.run`` cannot be called while a trace is
   active — it submits its own graph.  The inner workflow's step is a compiled
   ``Executable`` built by :class:`~evox_etl.core.StdWorkflow`, so an
   ``evaluate(candidates)`` closure traced by an OUTER ``StdWorkflow`` can never
   call it; ETL also has no equivalent of ``torch.func.stack_module_state`` nor of
   the ``register_vmap`` custom op that maps the inner state over the outer
   population.  A faithful in-graph HPO is therefore NOT portable, and this module
   does not pretend otherwise.
2. **What is implemented instead — a real, working host-side HPO.**
   :class:`HPOProblemWrapper` holds the inner algorithm/problem configs, the
   hyperparameter spec (which inner-config fields are tuned + how a candidate
   vector maps onto them), the generation count and the repeat policy.  Its
   :meth:`HPOProblemWrapper.evaluate` is plain host Python: substitute the
   candidate values into the inner configs (``dataclasses.replace``), build a fresh
   inner ``StdWorkflow`` HOST-side, run it (``wf.run(generations, seed=...)``) and
   read the inner monitor's best fitness (``wf.monitor.get_best_fitness()`` — the
   torch ``HPOMonitor.tell_fitness`` analogue), aggregated over ``num_repeats``
   seeds.  :func:`random_search` is a minimal numpy search driver making the
   wrapper runnable end-to-end.

## Limitations (precise)

- ``HPOProblemWrapper`` is NOT an ``evox_etl`` ``Problem``: it cannot be passed to
  a ``StdWorkflow`` as the outer problem.  The outer loop is a host-side search
  (:func:`random_search`, or any external optimizer calling :meth:`evaluate`).
- No batched instances: torch's ``num_instances`` + ``vmap`` parallelism has no
  analogue; candidates are evaluated one at a time
  (:meth:`HPOProblemWrapper.evaluate_many` is a plain Python loop).
- Hyperparameters must be plain STATIC config values (Python scalars, flat float
  tuples, strings) that ``dataclasses.replace`` can substitute; tensor-valued or
  otherwise traced hyperparameters are unsupported, and only fields of the inner
  algorithm/problem/monitor configs can be tuned.
- Every distinct candidate re-traces and re-compiles the inner step graph, because
  config values are baked into a trace as static leaves (``DESIGN.md`` §4.3), so
  throughput is dominated by per-candidate build time.  ``num_repeats`` reuses ONE
  built workflow per candidate (only the seed changes).
- The inner monitor must be an :class:`~evox_etl.workflows.EvalMonitorConfig`
  (``HPOFitnessMonitorConfig.inner_monitor``); a fresh copy with empty history
  containers is created per candidate so host-side history never accumulates.
  Multi-objective inner problems additionally need ``multi_obj_metric`` (the inner
  ``get_best_fitness`` accessor is single-objective only) plus
  ``full_fit_history=True``; ``multi_obj_metric`` is a HOST-side callable over a
  numpy ``(n_pareto, n_objs)`` array, so an etl metric function (e.g.
  ``evox_etl.metrics.igd``) must first be wrapped host-side with ``etl.build`` +
  ``etl.run`` — etl ops cannot run outside a trace.
- The torch plumbing ``HPOData`` / ``_hpo_evaluate_loop`` / ``copy_init_state`` has
  no analogue here: the inner workflow is rebuilt per candidate, so there is no
  shared init state to copy.
- Host-side module: numpy is used only for the search driver / bounds handling and
  this module must never be called from inside a trace.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from evox_etl.core import StdWorkflow
from evox_etl.workflows import EvalMonitorConfig

__all__ = [
    "HPSlot",
    "HPOFitnessMonitorConfig",
    "HPOMonitor",
    "HPOFitnessMonitor",
    "HPOProblemConfig",
    "HPOProblemWrapper",
    "random_search",
]

#: The inner configs a hyperparameter slot may patch, and the monitor config fields
#: holding host-side Python history containers.
CONFIG_KINDS: tuple[str, ...] = ("algorithm", "problem", "monitor")
_HISTORY_FIELDS: tuple[str, ...] = ("fit_history", "sol_history", "pop_history", "aux_history")

FitAggregation = str | Callable[[np.ndarray], float]


@dataclasses.dataclass(frozen=True)
class HPSlot:
    """One tunable field of an inner config (one candidate-vector position).

    :param config_kind: Which inner config the field lives in: ``"algorithm"``,
        ``"problem"`` or ``"monitor"``.
    :param field: The field name inside that config dataclass.
    :param transform: Optional host-side callable applied to the raw candidate
        value before substitution (e.g. ``lambda v: int(round(v))`` for an integer
        field).  Defaults to None (the raw value is passed through).
    """

    config_kind: str
    field: str
    transform: Callable[[float], Any] | None = None

    @property
    def name(self) -> str:
        """The dotted slot name (``"algorithm.w"``), used as the mapping key."""
        return f"{self.config_kind}.{self.field}"


@dataclasses.dataclass(frozen=True)
class HPOFitnessMonitorConfig:
    """Config of the inner-workflow fitness monitor used by HPO.

    The inner monitor itself is a reused :class:`EvalMonitorConfig` (best-so-far
    elite tracking); this dataclass only carries the HPO policy fields of the
    torch ``HPOFitnessMonitor``.

    :param inner_monitor: Config of the inner workflow's EvalMonitor. Defaults to
        a fresh :class:`EvalMonitorConfig`.
    :param fit_aggregation: How per-repeat best-fitness values are combined:
        ``"mean"`` (default), ``"min"``, ``"max"``, or a callable
        ``numpy.ndarray -> float``.
    :param multi_obj_metric: Multi-objective metric (e.g. ``evox_etl.metrics.hv``)
        applied to the inner Pareto-front fitness; required for multi-objective
        inner problems, unused otherwise. Defaults to None.
    """

    inner_monitor: EvalMonitorConfig = dataclasses.field(default_factory=EvalMonitorConfig)
    fit_aggregation: FitAggregation = "mean"
    multi_obj_metric: Callable[[np.ndarray], np.ndarray] | None = None


def _check_fit_aggregation(fit_aggregation: FitAggregation) -> None:
    """Validate a fit-aggregation policy (string choices or a callable)."""
    if callable(fit_aggregation):
        return
    if fit_aggregation not in ("mean", "min", "max"):
        raise ValueError(f"fit_aggregation must be callable or one of 'mean', 'min', 'max', got {fit_aggregation!r}")


class HPOMonitor:
    """Host-side base of the HPO monitors (torch ``HPOMonitor`` analogue).

    Unlike torch's ``tell_fitness()``, :meth:`tell_fitness` takes the inner
    workflow to read from, since the host-side redesign keeps no state pointer.
    """

    def __init__(self, num_repeats: int = 1, fit_aggregation: FitAggregation = "mean") -> None:
        """Store the repeat count and the per-repeat aggregation policy."""
        if num_repeats < 1:
            raise ValueError(f"num_repeats should be greater than 0, got {num_repeats}")
        _check_fit_aggregation(fit_aggregation)
        self.num_repeats = num_repeats
        self.fit_aggregation = fit_aggregation

    def tell_fitness(self, workflow: StdWorkflow) -> float:
        """Best fitness found by the last inner run (torch ``tell_fitness`` analogue)."""
        raise NotImplementedError("`tell_fitness` is not implemented; subclass HPOMonitor.")

    def aggregate(self, fitness: Sequence[float]) -> float:
        """Combine the per-repeat best-fitness values into one scalar."""
        values = np.asarray(fitness, dtype=np.float64).reshape(-1)
        if values.size == 0:
            raise ValueError("Cannot aggregate an empty sequence of fitness values")
        if callable(self.fit_aggregation):
            return float(self.fit_aggregation(values))
        if self.fit_aggregation == "mean":
            return float(np.mean(values))
        if self.fit_aggregation == "min":
            return float(np.min(values))
        if self.fit_aggregation == "max":
            return float(np.max(values))
        raise ValueError(f"fit_aggregation must be callable or one of 'mean', 'min', 'max', got {self.fit_aggregation!r}")


class HPOFitnessMonitor(HPOMonitor):
    """Reads the best fitness of a finished inner workflow (torch ``HPOFitnessMonitor`` analogue)."""

    def __init__(
        self,
        num_repeats: int = 1,
        fit_aggregation: FitAggregation = "mean",
        multi_obj_metric: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        """Store the repeat/aggregation policy and the optional MO metric."""
        super().__init__(num_repeats, fit_aggregation)
        if multi_obj_metric is not None and not callable(multi_obj_metric):
            raise ValueError(f"multi_obj_metric must be None or callable, got {multi_obj_metric!r}")
        self.multi_obj_metric = multi_obj_metric

    def tell_fitness(self, workflow: StdWorkflow) -> float:
        """Best fitness of the inner workflow's last ``run`` (min semantics).

        Single-objective: the inner monitor's best-so-far fitness.  With
        ``multi_obj_metric`` set, the metric is applied to the inner Pareto-front
        fitness and its minimum returned (torch ``HPOFitnessMonitor`` parity).
        """
        monitor = getattr(workflow, "monitor", None)
        if monitor is None or not hasattr(monitor, "get_best_fitness"):
            raise ValueError(
                "HPO requires the inner workflow to be built with an EvalMonitorConfig "
                f"(host-side EvalMonitor), got monitor={monitor!r}"
            )
        if self.multi_obj_metric is None:
            return float(monitor.get_best_fitness())
        pareto_fitness = np.asarray(monitor.get_pf_fitness(), dtype=np.float64)
        return float(np.min(np.asarray(self.multi_obj_metric(pareto_fitness))))


@dataclasses.dataclass(frozen=True)
class HPOProblemConfig:
    """Frozen config of the host-side HPO problem.

    :param algorithm: Inner algorithm config dataclass.
    :param problem: Inner problem config dataclass.
    :param hyperparameters: Slots declaring which inner-config fields are tuned;
        candidate-vector positions follow this order.
    :param num_generations: Generations the inner workflow runs per candidate.
    :param monitor: HPO monitor config (inner EvalMonitorConfig + HPO policy).
    :param num_repeats: Independent inner runs per candidate; repeat ``r`` uses seed
        ``seed + r``, and the per-repeat best fitness values are aggregated together.
    :param seed: Base seed of the inner runs. Defaults to 42.
    :param opt_direction: Opt direction of the inner workflow (a list for
        multi-objective). Defaults to "min".
    :param backend: ETL backend of the inner workflow. Defaults to "numpy".
    :param device: Device of the inner workflow. Defaults to None (cpu).
    """

    algorithm: Any
    problem: Any
    hyperparameters: tuple[HPSlot, ...] = ()
    num_generations: int = 1
    monitor: HPOFitnessMonitorConfig = dataclasses.field(default_factory=HPOFitnessMonitorConfig)
    num_repeats: int = 1
    seed: int = 42
    opt_direction: str | list[str] = "min"
    backend: str = "numpy"
    device: str | None = None


def _validate_config(config: HPOProblemConfig) -> HPOProblemConfig:
    """Validate an :class:`HPOProblemConfig` (ValueError, no bare asserts)."""
    if not isinstance(config, HPOProblemConfig):
        raise TypeError(f"config must be an HPOProblemConfig, got {type(config)}")
    for label, inner in (("algorithm", config.algorithm), ("problem", config.problem)):
        if not dataclasses.is_dataclass(inner) or isinstance(inner, type):
            raise ValueError(f"The inner {label} must be an INSTANCE of a config dataclass, got {inner!r}")
    if not isinstance(config.monitor, HPOFitnessMonitorConfig):
        raise TypeError(f"monitor must be an HPOFitnessMonitorConfig, got {type(config.monitor)}")
    if not dataclasses.is_dataclass(config.monitor.inner_monitor):
        raise ValueError(f"monitor.inner_monitor must be an EvalMonitorConfig instance, got {config.monitor.inner_monitor!r}")
    _check_fit_aggregation(config.monitor.fit_aggregation)
    for label, count in (("num_generations", config.num_generations), ("num_repeats", config.num_repeats)):
        if count < 1:
            raise ValueError(f"{label} should be greater than 0, got {count}")
    if not isinstance(config.seed, int):
        raise ValueError(f"seed must be an int, got {config.seed!r}")

    seen: set[str] = set()
    for slot in config.hyperparameters:
        if not isinstance(slot, HPSlot):
            raise TypeError(f"hyperparameters must contain HPSlot entries, got {slot!r}")
        if slot.config_kind not in CONFIG_KINDS:
            raise ValueError(f"HPSlot config_kind must be one of {CONFIG_KINDS}, got {slot.config_kind!r}")
        if slot.transform is not None and not callable(slot.transform):
            raise ValueError(f"HPSlot transform must be None or callable, got {slot.transform!r}")
        inner = _inner_config(config, slot.config_kind)
        field_names = {f.name for f in dataclasses.fields(inner)}
        if slot.field not in field_names:
            raise ValueError(
                f"HPSlot field {slot.field!r} is not a field of the inner {slot.config_kind} "
                f"config {type(inner).__name__}; available fields: {sorted(field_names)}"
            )
        if slot.name in seen:
            raise ValueError(f"Duplicate hyperparameter slot {slot.name!r}")
        seen.add(slot.name)
    return config


def _inner_config(config: HPOProblemConfig, kind: str) -> Any:
    """The inner config a slot of `kind` patches."""
    inner = {
        "algorithm": config.algorithm,
        "problem": config.problem,
        "monitor": config.monitor.inner_monitor,
    }.get(kind)
    if inner is None:
        raise ValueError(f"Unknown config kind {kind!r}, expected one of {CONFIG_KINDS}")
    return inner


def _fresh_inner_monitor(config: EvalMonitorConfig) -> EvalMonitorConfig:
    """Copy an inner monitor config with FRESH (empty) host-side history containers."""
    field_names = {f.name for f in dataclasses.fields(config)}
    fresh = {name: {} if name == "aux_history" else [] for name in _HISTORY_FIELDS if name in field_names}
    return dataclasses.replace(config, **fresh) if fresh else config


class HPOProblemWrapper:
    """Host-side HPO problem: evaluates hyperparameters by running an inner workflow.

    See the module docstring "Known Issues" — this is deliberately NOT an
    ``evox_etl`` ``Problem`` (nested compiled workflows cannot run inside an outer
    trace); the outer search is a host-side loop driving :meth:`evaluate`.

    Example:
    ```
    hpo = HPOProblemWrapper(HPOProblemConfig(
        algorithm=PSO(pop_size=30, lb=(-5.12,) * 5, ub=(5.12,) * 5),
        problem=Sphere(),
        hyperparameters=(HPSlot("algorithm", "w"), HPSlot("algorithm", "phi_p")),
        num_generations=30,
    ))
    fitness = hpo.evaluate([0.4, 1.0])            # or {"algorithm.w": 0.4, ...}
    ```
    """

    def __init__(self, config: HPOProblemConfig) -> None:
        """Validate `config` (ValueError on bad input) and prepare the monitor."""
        self.config = _validate_config(config)
        self.monitor = HPOFitnessMonitor(
            num_repeats=self.config.num_repeats,
            fit_aggregation=self.config.monitor.fit_aggregation,
            multi_obj_metric=self.config.monitor.multi_obj_metric,
        )
        self._slots = {slot.name: slot for slot in self.config.hyperparameters}
        self._slot_names = tuple(self._slots)
        self._defaults = {
            slot.name: getattr(_inner_config(self.config, slot.config_kind), slot.field) for slot in self.config.hyperparameters
        }

    # ------------------------------------------------------------- accessors

    @property
    def num_hyperparameters(self) -> int:
        """Length of the candidate vector, i.e. the number of tuned fields."""
        return len(self.config.hyperparameters)

    def get_params_keys(self) -> list[str]:
        """Dotted names of the tunable hyperparameters, in candidate order."""
        return list(self._slot_names)

    def get_init_params(self) -> dict[str, float]:
        """Current (default) values of the tunable hyperparameters (torch parity)."""
        return dict(self._defaults)

    # ------------------------------------------------------------- evaluation

    def evaluate(self, hyperparams: Mapping[str, Any] | Sequence[float] | None = None) -> float:
        """Fitness of one hyperparameter candidate (inner best fitness, min semantics).

        :param hyperparams: Either a mapping keyed by the :meth:`get_params_keys`
            names (missing keys keep the inner config's current values — torch's
            ``evaluate({})`` behaviour) or a vector of raw values in slot order.
            None (or ``{}``) evaluates the inner configs unchanged. Defaults to None.
        :return: The aggregated inner best fitness (a Python float).
        """
        values = self._resolve_values(hyperparams)
        workflow = self._build_workflow(values)
        fitness: list[float] = []
        for repeat in range(self.config.num_repeats):
            workflow.run(self.config.num_generations, seed=self.config.seed + repeat)
            fitness.append(self.monitor.tell_fitness(workflow))
        return self.monitor.aggregate(fitness)

    def evaluate_many(self, candidates: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        """Evaluate an ``(n, num_hyperparameters)`` batch of candidates (host-side loop)."""
        array = np.asarray(candidates, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] != self.num_hyperparameters:
            raise ValueError(f"candidates must have shape (n, {self.num_hyperparameters}), got {array.shape}")
        return np.asarray([self.evaluate(row) for row in array], dtype=np.float64)

    # -------------------------------------------------------------- internals

    def _resolve_values(self, hyperparams: Mapping[str, Any] | Sequence[float] | None) -> dict[str, Any]:
        """Map a candidate onto ``{slot name: value}``, applying slot transforms."""
        if isinstance(hyperparams, (str, bytes)):
            raise TypeError("hyperparams must be a mapping, a vector or None")
        if hyperparams is None:
            raw: dict[str, Any] = dict(self._defaults)
        elif isinstance(hyperparams, Mapping):
            unknown = [key for key in hyperparams if key not in self._slots]
            if unknown:
                raise ValueError(f"Unknown hyperparameter(s) {sorted(unknown)}; available keys are {self.get_params_keys()}")
            raw = {**self._defaults, **dict(hyperparams)}
        else:
            vector = np.asarray(hyperparams, dtype=np.float64).reshape(-1)
            if vector.shape[0] != self.num_hyperparameters:
                raise ValueError(f"Expected {self.num_hyperparameters} hyperparameter value(s), got {vector.shape[0]}")
            if not np.all(np.isfinite(vector)):
                raise ValueError(f"Hyperparameter values must be finite, got {vector}")
            raw = {name: float(value) for name, value in zip(self._slot_names, vector)}
        return {
            slot.name: (slot.transform(raw[slot.name]) if slot.transform is not None else raw[slot.name])
            for slot in self.config.hyperparameters
        }

    def _build_workflow(self, values: Mapping[str, Any]) -> StdWorkflow:
        """Build a fresh inner ``StdWorkflow`` with the candidate values substituted."""
        algorithm, problem = self.config.algorithm, self.config.problem
        monitor = _fresh_inner_monitor(self.config.monitor.inner_monitor)
        for slot in self.config.hyperparameters:
            value = values[slot.name]
            if slot.config_kind == "algorithm":
                algorithm = dataclasses.replace(algorithm, **{slot.field: value})
            elif slot.config_kind == "problem":
                problem = dataclasses.replace(problem, **{slot.field: value})
            else:
                monitor = dataclasses.replace(monitor, **{slot.field: value})
        return StdWorkflow(
            algorithm=algorithm,
            problem=problem,
            monitor=monitor,
            opt_direction=self.config.opt_direction,
            num_generations=self.config.num_generations,
            backend=self.config.backend,
            device=self.config.device,
        )


def _broadcast_bounds(
    bounds: Sequence[float] | Sequence[Sequence[float]], num_hyperparameters: int
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize search bounds to ``(low, high)`` vectors of length `num_hyperparameters`.

    Accepts one ``(low, high)`` pair broadcast to every slot, or one pair per slot.
    """
    pairs = list(bounds)
    if len(pairs) == 2 and all(np.isscalar(entry) for entry in pairs):
        low = np.full(num_hyperparameters, float(pairs[0]))
        high = np.full(num_hyperparameters, float(pairs[1]))
    elif len(pairs) == num_hyperparameters and all(len(entry) == 2 for entry in pairs):
        low = np.asarray([float(entry[0]) for entry in pairs])
        high = np.asarray([float(entry[1]) for entry in pairs])
    else:
        raise ValueError(
            f"bounds must be one (low, high) pair applied to every slot or "
            f"{num_hyperparameters} (low, high) pairs, got {pairs!r}"
        )
    if not np.all(low < high):
        raise ValueError(f"Every bound must satisfy low < high, got low={low}, high={high}")
    return low, high


def random_search(
    wrapper: HPOProblemWrapper,
    bounds: Sequence[float] | Sequence[Sequence[float]],
    n_trials: int = 20,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    """Uniform-random host-side search over the wrapper's hyperparameter box.

    :param wrapper: An :class:`HPOProblemWrapper` to evaluate candidates with.
    :param bounds: One ``(low, high)`` pair broadcast to all slots, or one pair per slot.
    :param n_trials: Number of candidates to sample and evaluate (must be > 0).
    :param seed: Seed of the host numpy generator. Defaults to 42.
    :return: ``(best_hyperparams, best_fitness)`` — the best raw candidate vector
        (shape ``(num_hyperparameters,)``) and its fitness (minimization).
    """
    if n_trials < 1:
        raise ValueError(f"n_trials should be greater than 0, got {n_trials}")
    low, high = _broadcast_bounds(bounds, wrapper.num_hyperparameters)
    rng = np.random.default_rng(seed)
    candidates = rng.uniform(low, high, size=(n_trials, wrapper.num_hyperparameters))
    fitness = wrapper.evaluate_many(candidates)
    best_index = int(np.argmin(fitness))
    return candidates[best_index], float(fitness[best_index])

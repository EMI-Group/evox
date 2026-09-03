"""StdWorkflow: composes algorithm/problem/monitor functions into one compiled step graph."""

from __future__ import annotations

import dataclasses
import importlib
from typing import Any, Callable

import etl
import etl.core
import etl.random
import numpy as np

from .state import replace

__all__ = ["EmptyState", "WorkflowState", "StdWorkflow"]


@dataclasses.dataclass(frozen=True)
class EmptyState:
    """Empty state for stateless components (e.g. numerical problems)."""


@dataclasses.dataclass(frozen=True)
class WorkflowState:
    """State of a running workflow.

    All leaves are ETL tensors: `generation` is a 0-d int32 tensor and `key` a
    0-d int64 tensor. `monitor_state` is a 0-d float32 dummy tensor when no
    monitor is configured.
    """

    algorithm_state: Any
    problem_state: Any
    monitor_state: Any
    generation: Any
    key: Any


def _module_of(config: Any) -> Any:
    """Resolve the module a config dataclass was defined in (function convention)."""
    return importlib.import_module(type(config).__module__)


def _parse_device(device: str | etl.core.Device | None) -> etl.core.Device:
    """Parse a device spec: None -> cpu, str ("cpu"/"cuda"/"cuda:N") or Device."""
    if device is None:
        return etl.core.Device("cpu")
    if isinstance(device, etl.core.Device):
        return device
    if isinstance(device, str):
        text = device.strip().lower()
        if ":" in text:
            kind, _, index = text.partition(":")
            return etl.core.Device(kind, int(index))
        return etl.core.Device(text)
    raise TypeError(f"device must be None, a str or an etl.core.Device, got {type(device)}")


def _parse_opt_direction(opt_direction: str | list[str] | tuple[str, ...]) -> tuple[int, ...]:
    """Validate "min"/"max" (or a list thereof) and return a tuple of 1/-1 ints."""
    if isinstance(opt_direction, str):
        if opt_direction not in ("min", "max"):
            raise ValueError(f"Expect optimization direction to be `min` or `max`, got {opt_direction}")
        return (1 if opt_direction == "min" else -1,)
    if isinstance(opt_direction, (list, tuple)):
        if not opt_direction or not all(d in ("min", "max") for d in opt_direction):
            raise ValueError(f"Expect optimization direction to be `min` or `max`, got {opt_direction}")
        return tuple(1 if d == "min" else -1 for d in opt_direction)
    raise TypeError(
        f"opt_direction must be a str or a list of str (`min`/`max`), got {type(opt_direction)}"
    )


def _dummy_monitor_state() -> Any:
    """0-d float32 constant placeholder for a missing monitor (baked in-graph)."""
    return etl.ops.constant(etl.core.tensor(np.asarray(0.0, dtype=np.float32)))


def _zero_generation() -> Any:
    """0-d int32 constant used as the initial generation counter."""
    return etl.cast(etl.ops.constant(etl.core.tensor(np.asarray(0, dtype=np.int64))), np.int32)


def _opt_direction_constant(opt_dir: tuple[int, ...]) -> Any:
    """Bake the opt direction as an in-graph float32 constant: () or (m,)."""
    arr = np.asarray(opt_dir, dtype=np.float32)
    if arr.ndim == 1 and arr.shape[0] == 1:
        arr = arr.reshape(())
    return etl.ops.constant(etl.core.tensor(arr))


class StdWorkflow:
    """The standard optimization workflow (functional).

    Composes the plain functions of an algorithm, a problem and (optionally) a
    monitor module into a single compiled step graph and runs it in a
    same-device loop.

    Usage:
    ```
    wf = StdWorkflow(PSO(...), Sphere(), monitor=EvalMonitorConfig(), opt_direction="min")
    state = wf.init(seed=42)
    for _ in range(20):
        state = wf.step(state)
    ```
    """

    def __init__(
        self,
        algorithm: Any,
        problem: Any,
        monitor: Any = None,
        opt_direction: str | list[str] = "min",
        solution_transform: Callable[[Any], Any] | None = None,
        fitness_transform: Callable[[Any], Any] | None = None,
        num_generations: int | None = None,
        backend: str = "numpy",
        device: str | etl.core.Device | None = None,
        compile_options: dict[str, Any] | None = None,
    ):
        """Initialize the workflow with static arguments.

        :param algorithm: Config dataclass of the algorithm.
        :param problem: Config dataclass of the problem.
        :param monitor: Config dataclass of the monitor, or None. Defaults to None.
        :param opt_direction: "min"/"max" (or a list of them for multi-objective). Defaults to "min".
        :param solution_transform: Plain callable applied to candidates before evaluation. Defaults to None (identity).
        :param fitness_transform: Plain callable applied to the opt-direction-scaled fitness. Defaults to None (identity).
        :param num_generations: Default number of generations for `run`/`fit`. Defaults to None.
        :param backend: etl compiler backend. Defaults to "numpy".
        :param device: Device to run the step graph on. Defaults to None (cpu).
        :param compile_options: Extra keyword options passed to `etl.build`. Defaults to None.
        """
        if not dataclasses.is_dataclass(algorithm):
            raise TypeError(f"algorithm must be a config dataclass, got {type(algorithm)}")
        if not dataclasses.is_dataclass(problem):
            raise TypeError(f"problem must be a config dataclass, got {type(problem)}")
        if monitor is not None and not dataclasses.is_dataclass(monitor):
            raise TypeError(f"monitor must be a config dataclass or None, got {type(monitor)}")
        if solution_transform is not None and not callable(solution_transform):
            raise TypeError(f"Expect solution transform to be callable, got {solution_transform}")
        if fitness_transform is not None and not callable(fitness_transform):
            raise TypeError(f"Expect fitness transform to be callable, got {fitness_transform}")

        self.algorithm = algorithm
        self.problem = problem
        self._monitor_cfg = monitor
        self._opt_direction = _parse_opt_direction(opt_direction)
        self._opt_direction_is_list = isinstance(opt_direction, (list, tuple))
        self.solution_transform = solution_transform
        self.fitness_transform = fitness_transform
        self.num_generations = num_generations
        self.backend = backend
        self._device = _parse_device(device)
        self._compile_options = dict(compile_options) if compile_options else {}

        self._algo_mod = _module_of(algorithm)
        self._prob_mod = _module_of(problem)
        self._mon_mod = _module_of(monitor) if monitor is not None else None

        # generation-0 branch: used only if BOTH init_ask and init_tell exist
        self._use_init_branch = (
            getattr(self._algo_mod, "init_ask", None) is not None
            and getattr(self._algo_mod, "init_tell", None) is not None
        )

        self.monitor = None  # host-side convenience wrapper (if the module provides one)
        self.monitor_config: Any = None  # completed monitor config
        self.monitor_state: Any = None  # latest monitor state
        self._state: WorkflowState | None = None
        self._init_exe = None
        self._mon_init_exe = None
        self._step_exe = None

    @property
    def opt_direction(self) -> tuple[int, ...]:
        """The optimization direction as a tuple of 1/-1 Python ints."""
        return self._opt_direction

    # ------------------------------------------------------------------ init

    def init(self, seed: int = 42) -> WorkflowState:
        """Build the init/step graphs (once) and return the initial workflow state."""
        if self._init_exe is None:
            self._init_exe = self._build_algorithm_init_exe()
        state = etl.run(self._init_exe, etl.random.key(seed))
        return self._finish_init(state)

    def _build_algorithm_init_exe(self) -> Any:
        """Build the init graph (numpy backend, cpu): algo init (+ optional problem init)."""
        algo_mod, algo_cfg = self._algo_mod, self.algorithm
        prob_mod, prob_cfg = self._prob_mod, self.problem
        algo_init = algo_mod.init
        prob_init = getattr(prob_mod, "init", None)

        def body(key: Any) -> WorkflowState:
            key, subkey = etl.random.split(key)
            alg_state = algo_init(algo_cfg, subkey)
            if prob_init is not None:
                out = prob_init(prob_cfg, subkey)
                if type(out) is tuple and len(out) == 2:
                    # problem init may return (prob_state, advanced_key)
                    prob_state, key = out
                else:
                    prob_state = out
            else:
                prob_state = EmptyState()
            return WorkflowState(
                algorithm_state=alg_state,
                problem_state=prob_state,
                monitor_state=_dummy_monitor_state(),  # replaced host-side when a monitor is set
                generation=_zero_generation(),
                key=key,
            )

        key_spec = etl.core.TensorSpec((), np.int64)
        return etl.build(body, key_spec, backend="numpy", device=etl.core.Device("cpu"))

    def _finish_init(self, state: WorkflowState) -> WorkflowState:
        """Host-side part of init: monitor setup, device placement, step graph build."""
        if self._mon_mod is not None:
            if self.monitor_config is None:
                pop_size, dim = self._discover_pop_size(state)
                self.monitor_config = self._complete_monitor_config(pop_size, dim)
            state = self._run_monitor_init(state)
            self.monitor_state = state.monitor_state
            self._setup_monitor_wrapper()
        if self._device.kind != "cpu":
            state = etl.tree_map(lambda t: t.to(self._device), state)
        self._state = state
        if self._step_exe is None:
            self._step_exe = self._build_step_exe(state)
        return state

    def _discover_pop_size(self, state: WorkflowState) -> tuple[int | None, int | None]:
        """Infer (pop_size, dim) from the algorithm state, falling back to config fields."""
        alg_state = state.algorithm_state
        population = getattr(alg_state, "population", None)
        if population is None:
            # PSO-style states store the population as `pop` (no `or` on tensors).
            population = getattr(alg_state, "pop", None)
        shape = getattr(population, "shape", None)
        if shape is not None and len(shape) >= 2:
            return int(shape[0]), int(shape[1])
        cfg = self.algorithm
        pop_size = int(shape[0]) if shape is not None and len(shape) == 1 else getattr(cfg, "pop_size", None)
        dim = getattr(cfg, "dim", None)
        return pop_size, dim

    def _complete_monitor_config(self, pop_size: int | None, dim: int | None) -> Any:
        """Fill monitor config fields the workflow can infer (pop_size/dim/n_obj/multi_obj)."""
        cfg = self._monitor_cfg
        field_names = {f.name for f in dataclasses.fields(cfg)}
        changes: dict[str, Any] = {}
        if "pop_size" in field_names and pop_size is not None:
            changes["pop_size"] = pop_size
        if "dim" in field_names and dim is not None:
            changes["dim"] = dim
        if "n_obj" in field_names and self._opt_direction_is_list:
            changes["n_obj"] = len(self._opt_direction)
        if "multi_obj" in field_names and getattr(cfg, "multi_obj", None) is None:
            changes["multi_obj"] = len(self._opt_direction) > 1
        if "opt_direction" in field_names:
            # the workflow's direction is authoritative (monitor un-negates with it)
            changes["opt_direction"] = self._opt_direction
        if not changes:
            return cfg
        return dataclasses.replace(cfg, **changes)

    def _run_monitor_init(self, state: WorkflowState) -> WorkflowState:
        """Run the monitor init graph (numpy/cpu) and swap its state into the workflow state."""
        mon_init = getattr(self._mon_mod, "init", None)
        if mon_init is None:
            return replace(state, monitor_state=EmptyState())
        if self._mon_init_exe is None:
            self._mon_init_exe = self._build_monitor_init_exe(mon_init)
        mon_state, key = etl.run(self._mon_init_exe, state.key)
        return replace(state, monitor_state=mon_state, key=key)

    def _build_monitor_init_exe(self, mon_init: Callable[..., Any]) -> Any:
        """Build a tiny init graph: split key, mon_init(config, subkey) -> (state, key)."""
        mon_cfg = self.monitor_config

        def body(key: Any) -> tuple[Any, Any]:
            key, subkey = etl.random.split(key)
            return mon_init(mon_cfg, subkey), key

        key_spec = etl.core.TensorSpec((), np.int64)
        return etl.build(body, key_spec, backend="numpy", device=etl.core.Device("cpu"))

    def _setup_monitor_wrapper(self) -> None:
        """Create the host-side monitor convenience wrapper, if the module provides one."""
        wrapper_cls = getattr(self._mon_mod, "Monitor", None)
        if wrapper_cls is not None:
            self.monitor = wrapper_cls(config=self.monitor_config, state=self.monitor_state)

    # ------------------------------------------------------------------ step

    def step(self, state: WorkflowState | None = None) -> WorkflowState:
        """Perform a single optimization step (one compiled graph run)."""
        if state is None:
            state = self._state
        if state is None:
            raise RuntimeError("Workflow is not initialized: call init() first.")
        if self._device.kind != "cpu":
            state = etl.tree_map(lambda t: t.to(self._device), state)
        if self._step_exe is None:
            self._step_exe = self._build_step_exe(state)
        state = etl.run(self._step_exe, state)
        self._record_history(state)
        self._state = state
        return state

    def _build_step_exe(self, state: WorkflowState) -> Any:
        """Build the step graph ONCE from tensor specs of the current state."""
        specs = etl.tree_map(
            lambda t: etl.core.TensorSpec(tuple(t.shape), t.dtype),
            state,
        )
        step_fn = self._make_step_fn()
        options = {k: v for k, v in self._compile_options.items() if k not in ("backend", "device")}
        return etl.build(step_fn, specs, backend=self.backend, device=self._device, **options)

    def _make_step_fn(self) -> Callable[[WorkflowState], WorkflowState]:
        """Compose the full step: ask -> evaluate -> transform -> tell -> monitor, gen+1."""
        algo_mod, algo_cfg = self._algo_mod, self.algorithm
        prob_mod, prob_cfg = self._prob_mod, self.problem
        mon_mod, mon_cfg = self._mon_mod, self.monitor_config
        ask = algo_mod.ask
        tell = algo_mod.tell
        init_ask = getattr(algo_mod, "init_ask", None)
        init_tell = getattr(algo_mod, "init_tell", None)
        evaluate = prob_mod.evaluate
        mon_update = getattr(mon_mod, "monitor_update", None) if mon_mod is not None else None
        solution_transform = self.solution_transform
        fitness_transform = self.fitness_transform
        opt_dir = self._opt_direction
        use_init_branch = self._use_init_branch

        def step_fn(state: WorkflowState) -> WorkflowState:
            opt_dir_const = _opt_direction_constant(opt_dir)

            def run_generation(st: WorkflowState, first: bool) -> WorkflowState:
                if first:
                    candidates, alg_state = init_ask(algo_cfg, st.algorithm_state)
                else:
                    candidates, alg_state = ask(algo_cfg, st.algorithm_state)
                # monitor sees RAW candidates (post_ask semantics)
                if solution_transform is not None:
                    x = solution_transform(candidates)
                else:
                    x = candidates
                fitness, prob_state = evaluate(prob_cfg, st.problem_state, x)
                fitness = opt_dir_const * fitness
                if fitness_transform is not None:
                    fitness = fitness_transform(fitness)
                # monitor sees TRANSFORMED fitness (pre_tell semantics)
                if mon_update is not None:
                    mon_state = mon_update(mon_cfg, st.monitor_state, candidates, fitness)
                else:
                    mon_state = st.monitor_state
                if first:
                    alg_state = init_tell(algo_cfg, alg_state, fitness)
                else:
                    alg_state = tell(algo_cfg, alg_state, fitness)
                return WorkflowState(
                    algorithm_state=alg_state,
                    problem_state=prob_state,
                    monitor_state=mon_state,
                    generation=etl.cast(st.generation + 1, np.int32),
                    key=st.key,
                )

            def first_branch(st: WorkflowState) -> WorkflowState:
                return run_generation(st, True)

            def regular_branch(st: WorkflowState) -> WorkflowState:
                return run_generation(st, False)

            if use_init_branch:
                return etl.cond(state.generation == 0, first_branch, regular_branch, state)
            return regular_branch(state)

        return step_fn

    def _record_history(self, state: WorkflowState) -> None:
        """Host-side history recording: copy monitor state leaves to the config's lists."""
        if self.monitor is not None:
            self.monitor.state = state.monitor_state
        cfg = self.monitor_config
        if cfg is None:
            return
        field_names = {f.name for f in dataclasses.fields(cfg)}
        has_history = any(
            name in field_names for name in ("fit_history", "sol_history", "pop_history")
        )
        if not has_history:
            return
        mon_state = state.monitor_state
        cpu = etl.core.Device("cpu")
        if "fit_history" in field_names and getattr(cfg, "full_fit_history", False):
            latest_fitness = getattr(mon_state, "latest_fitness", None)
            if latest_fitness is not None:
                getattr(cfg, "fit_history").append(latest_fitness.to(cpu).numpy())
        if "sol_history" in field_names and getattr(cfg, "full_sol_history", False):
            latest_solution = getattr(mon_state, "latest_solution", None)
            if latest_solution is not None:
                getattr(cfg, "sol_history").append(latest_solution.to(cpu).numpy())
        if "pop_history" in field_names and getattr(cfg, "full_pop_history", False):
            latest_solution = getattr(mon_state, "latest_solution", None)
            if latest_solution is not None:
                getattr(cfg, "pop_history").append(latest_solution.to(cpu).numpy())

    # ------------------------------------------------------------------ run

    def run(self, generations: int | None = None, seed: int = 42) -> WorkflowState:
        """Initialize and run the optimization loop for `generations` generations.

        If `generations` is None, `self.num_generations` is used; if both are
        None, a ValueError is raised.
        """
        generations = self._resolve_generations(generations, "run")
        state = self.init(seed=seed)
        return self._run_loop(state, generations)

    def fit(self, fitness: Any = None, generations: int | None = None, seed: int = 42) -> Any:
        """Run the optimization and return the best fitness found (min semantics).

        Requires a monitor wrapper exposing `get_best_fitness` (the EvalMonitor).
        If `fitness` is given and the monitor config has a `fit_history` list,
        it is appended as the generation-0 entry before the loop.
        """
        generations = self._resolve_generations(generations, "fit")
        state = self.init(seed=seed)
        cfg = self.monitor_config
        if fitness is not None and cfg is not None:
            field_names = {f.name for f in dataclasses.fields(cfg)}
            if "fit_history" in field_names:
                getattr(cfg, "fit_history").append(np.asarray(fitness))
        self._run_loop(state, generations)
        if self.monitor is None or not hasattr(self.monitor, "get_best_fitness"):
            raise ValueError("fit requires an EvalMonitor (monitor with get_best_fitness)")
        return self.monitor.get_best_fitness()

    def _resolve_generations(self, generations: int | None, caller: str) -> int:
        """Resolve the generation count from the argument or `num_generations`."""
        generations = self.num_generations if generations is None else generations
        if generations is None:
            raise ValueError(
                f"The number of generations must be specified either in the constructor "
                f"(num_generations) or in {caller}(generations=...)."
            )
        return generations

    def _run_loop(self, state: WorkflowState, generations: int) -> WorkflowState:
        """Run `generations` compiled steps in a same-device loop."""
        for _ in range(generations):
            state = self.step(state)
        return state

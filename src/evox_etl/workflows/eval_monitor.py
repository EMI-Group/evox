"""Functional ETL port of ``src/evox/workflows/eval_monitor.py``.

Plain (non-defn) functions ``init``/``monitor_update`` live in the SAME module
as the frozen ``EvalMonitorConfig`` dataclass (the workflow resolves them via
``type(config).__module__``). The host-side ``EvalMonitor`` wrapper (alias
``Monitor``) exposes best-so-far / pareto-front accessors over the config's
history lists, which the workflow fills after each step.
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import field
from typing import Any

import etl
import etl.core
import numpy as np

from evox_etl.operators.selection import non_dominate_rank

__all__ = [
    "EvalMonitor",
    "EvalMonitorConfig",
    "EvalMonitorState",
    "MOEvalMonitorState",
    "Monitor",
]


@dataclasses.dataclass(frozen=True)
class EvalMonitorConfig:
    """Static config of the evaluation monitor (host-side history lists included).

    ``pop_size``/``dim``/``n_obj``/``multi_obj``/``opt_direction`` are completed
    by the workflow before ``init`` runs; the ``*_history`` lists are appended
    host-side by the workflow after each step (numpy arrays per generation).

    :param multi_obj: Whether the optimization is multi-objective. Defaults to None (auto-detected by the workflow).
    :param full_fit_history: Record every generation's fitness values. Defaults to True.
    :param full_sol_history: Record every generation's solutions. Defaults to False.
    :param full_pop_history: Record every generation's population. Defaults to False.
    :param topk: Number of elite solutions to track (single-objective only). Defaults to 1.
    :param pop_size: Population size (filled by the workflow). Defaults to None.
    :param dim: Solution dimension (filled by the workflow). Defaults to None.
    :param n_obj: Number of objectives (filled by the workflow). Defaults to None.
    :param opt_direction: Tuple of 1/-1 ints; stored fitness is scaled by it (filled by the workflow). Defaults to (1,).
    :param fit_history: Per-generation fitness values (filled host-side). Defaults to an empty list.
    :param sol_history: Per-generation solutions (filled host-side). Defaults to an empty list.
    :param pop_history: Per-generation populations (filled host-side). Defaults to an empty list.
    :param aux_history: Algorithm auxiliary tensors, keyed by name, mapped to a list of
        numpy arrays per generation (filled host-side by ``record_auxiliary``). Defaults to an empty dict.
    """

    multi_obj: bool | None = None
    full_fit_history: bool = True
    full_sol_history: bool = False
    full_pop_history: bool = False
    topk: int = 1
    pop_size: int | None = None
    dim: int | None = None
    n_obj: int | None = None
    opt_direction: tuple = (1,)
    fit_history: list = field(default_factory=list)
    sol_history: list = field(default_factory=list)
    pop_history: list = field(default_factory=list)
    aux_history: dict = field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class EvalMonitorState:
    """Single-objective monitor state; fixed-shape float32 tensor leaves only."""

    latest_solution: etl.SymbolicTensor
    latest_fitness: etl.SymbolicTensor
    topk_solutions: etl.SymbolicTensor
    topk_fitness: etl.SymbolicTensor


@dataclasses.dataclass(frozen=True)
class MOEvalMonitorState:
    """Multi-objective monitor state; latest solution/fitness float32 leaves only."""

    latest_solution: etl.SymbolicTensor
    latest_fitness: etl.SymbolicTensor


def _zeros(shape: tuple[int, ...]) -> etl.SymbolicTensor:
    """In-graph float32 zeros constant (top-level ``etl.zeros`` is eager-only)."""
    return etl.ops.constant(etl.core.tensor(np.zeros(shape, dtype=np.float32)))


def _inf(shape: tuple[int, ...]) -> etl.SymbolicTensor:
    """In-graph float32 +inf constant; initializes the SO elite so gen-0 always wins."""
    return etl.ops.constant(etl.core.tensor(np.full(shape, np.inf, dtype=np.float32)))


def _init_so(config: EvalMonitorConfig) -> EvalMonitorState:
    """Allocate the fixed single-objective buffers."""
    pop_size, dim, topk = config.pop_size, config.dim, config.topk
    if pop_size is None or dim is None:
        raise ValueError(
            "EvalMonitor requires pop_size and dim to initialize its buffers, "
            f"got pop_size={pop_size}, dim={dim}. The workflow fills them from the "
            "algorithm state (population/pop) or the algorithm config."
        )
    return EvalMonitorState(
        latest_solution=_zeros((pop_size, dim)),
        latest_fitness=_zeros((pop_size,)),
        topk_solutions=_zeros((topk, dim)),
        topk_fitness=_inf((topk,)),
    )


def _init_mo(config: EvalMonitorConfig) -> MOEvalMonitorState:
    """Allocate the fixed multi-objective buffers."""
    pop_size, dim, n_obj = config.pop_size, config.dim, config.n_obj
    if pop_size is None or dim is None or n_obj is None:
        raise ValueError(
            "EvalMonitor requires pop_size, dim and n_obj to initialize its buffers, "
            f"got pop_size={pop_size}, dim={dim}, n_obj={n_obj}. The workflow fills "
            "them from the algorithm state/config and the opt_direction list."
        )
    return MOEvalMonitorState(
        latest_solution=_zeros((pop_size, dim)),
        latest_fitness=_zeros((pop_size, n_obj)),
    )


def init(config: EvalMonitorConfig, key: etl.SymbolicTensor) -> Any:
    """Create the fixed-shape monitor state (``key`` unused, kept for the contract)."""
    if config.multi_obj:
        return _init_mo(config)
    return _init_so(config)


def monitor_update(
    config: EvalMonitorConfig,
    state: Any,
    candidate: etl.SymbolicTensor,
    fitness: etl.SymbolicTensor,
) -> Any:
    """One monitor update inside the step graph (static branch on fitness rank).

    Rank-1 fitness (SO): track the running top-``topk`` elite over the concat of
    the stored elite buffers and the new generation (monotone thanks to the
    +inf-initialized buffers, mirroring torch). Rank-2 fitness (MO): only record
    the latest solutions/fitness; the Pareto front is derived host-side.
    """
    if len(fitness.shape) == 1:
        # single-objective: running elite
        concat_fit = etl.concatenate([state.topk_fitness, fitness], axis=0)
        top_fit, indices = etl.topk(concat_fit, config.topk, axis=0, largest=False)
        concat_sol = etl.concatenate([state.topk_solutions, candidate], axis=0)
        topk_solutions = etl.gather(concat_sol, indices, axis=0)
        return EvalMonitorState(
            latest_solution=candidate,
            latest_fitness=fitness,
            topk_solutions=topk_solutions,
            topk_fitness=top_fit,
        )
    # multi-objective: history/pareto front are derived host-side
    return MOEvalMonitorState(latest_solution=candidate, latest_fitness=fitness)


def record_auxiliary(config: EvalMonitorConfig, aux: dict) -> None:
    """Host-side: record one generation of auxiliary tensors.

    Gated on ``config.full_pop_history`` (torch parity). Each value is a CONCRETE
    etl tensor, converted to numpy via ``.to(etl.core.Device('cpu')).numpy()``
    (mirroring the ``EvalMonitor._to_numpy`` idiom) and appended to
    ``config.aux_history[key]``.
    """
    if not config.full_pop_history:
        return
    for key, value in aux.items():
        arr = np.asarray(value.to(etl.core.Device("cpu")).numpy())
        config.aux_history.setdefault(key, []).append(arr)


# Byte-identical to the torch `EvalMonitor.plot` warning (torch parity).
_NO_VIS_TOOL_WARNING = 'No visualization tool available, return None. Hint: pip install "evox[vis]"'


def _vis_plot_module() -> Any:
    """The ``evox_etl.vis_tools.plot`` module, or None when plotting is unavailable.

    Plotly is an OPTIONAL dependency and ``evox_etl.vis_tools`` imports fine
    without it (its ``plot`` submodule only sets ``go = None``), so availability
    must be probed through both the import AND that module's ``go`` re-export.
    """
    try:
        import evox_etl.vis_tools.plot as vis_plot
    except ImportError:  # pragma: no cover - the package itself is always present
        return None
    if getattr(vis_plot, "go", None) is None:
        return None
    return vis_plot


class EvalMonitor:
    """Host-side evaluation monitor wrapper (workflow creates it as ``Monitor(config=..., state=...)``).

    Stored fitness is opt-direction-scaled (negated under ``opt_direction="max"``);
    all ``get_*fitness`` accessors undo that scaling. State leaves are concrete
    etl tensors converted to numpy arrays on access.
    """

    def __init__(self, config: EvalMonitorConfig, state: Any):
        """Wrap a monitor config and its current (post-step) state."""
        self.config = config
        self.state = state
        self._rank_exes: dict[tuple[int, ...], Any] = {}

    # ------------------------------------------------------------- helpers

    @staticmethod
    def _to_numpy(t: Any) -> np.ndarray:
        """Copy a concrete etl tensor leaf to a host numpy array."""
        return np.asarray(t.to(etl.core.Device("cpu")).numpy())

    def _unn(self, arr: np.ndarray) -> np.ndarray:
        """Undo the opt-direction scaling (a (1,) direction acts as a scalar)."""
        opt_dir = np.asarray(self.config.opt_direction, dtype=np.float32)
        if opt_dir.ndim == 1 and opt_dir.shape[0] == 1:
            opt_dir = opt_dir.reshape(())
        return arr * opt_dir

    def _non_dominate_rank(self, fitness: np.ndarray) -> np.ndarray:
        """0-based non-domination ranks, host-side via a lazily cached etl exe."""
        shape = tuple(fitness.shape)
        exe = self._rank_exes.get(shape)
        if exe is None:
            exe = etl.build(
                non_dominate_rank,
                etl.core.TensorSpec(shape, np.float32),
                backend="numpy",
                device=etl.core.Device("cpu"),
            )
            self._rank_exes[shape] = exe
        arr = np.asarray(fitness, dtype=np.float32)
        out = etl.run(exe, etl.core.tensor(arr))
        return np.asarray(out.numpy())

    def _check_single_objective(self) -> None:
        """Raise the torch-style error for SO-only accessors under MO."""
        if self.config.multi_obj:
            raise ValueError(
                "Multi-objective optimization does not have a single best solution. Please use get_pf_solutions"
            )

    # ------------------------------------------------------- latest/topk

    def get_latest_fitness(self) -> np.ndarray:
        """Get the fitness values of the latest generation (un-negated)."""
        return self._unn(self._to_numpy(self.state.latest_fitness))

    def get_latest_solution(self) -> np.ndarray:
        """Get the solutions of the latest generation."""
        return self._to_numpy(self.state.latest_solution)

    def get_topk_fitness(self) -> np.ndarray:
        """Get the top-k fitness values so far (un-negated)."""
        self._check_single_objective()
        return self._unn(self._to_numpy(self.state.topk_fitness))

    def get_topk_solutions(self) -> np.ndarray:
        """Get the top-k solutions so far."""
        self._check_single_objective()
        return self._to_numpy(self.state.topk_solutions)

    def get_best_fitness(self) -> float:
        """Get the best fitness value so far (un-negated)."""
        self._check_single_objective()
        return float(self._unn(self._to_numpy(self.state.topk_fitness))[0])

    def get_best_solution(self) -> np.ndarray:
        """Get the best solution so far."""
        self._check_single_objective()
        return self._to_numpy(self.state.topk_solutions)[0]

    # ------------------------------------------------------- pareto front

    def get_pf_fitness(self, deduplicate: bool = True) -> np.ndarray:
        """Approximate Pareto-front fitness over all evaluated solutions (un-negated).

        Requires ``full_fit_history``; deduplicates on fitness when requested.
        """
        if not self.config.multi_obj:
            raise ValueError("get_pf_fitness is only available for multi-objective optimization.")
        if not self.config.full_fit_history:
            warnings.warn("`get_pf_fitness` requires enabling `full_fit_history`.")
        all_fitness = np.concatenate(self.fitness_history, axis=0)
        if deduplicate:
            all_fitness = np.unique(all_fitness, axis=0)
        rank = self._non_dominate_rank(all_fitness)
        return self._unn(all_fitness[rank == 0])

    def get_pf_solutions(self, deduplicate: bool = True) -> np.ndarray:
        """Approximate Pareto-front solutions over all evaluated solutions.

        Requires ``full_fit_history`` and ``full_sol_history``; deduplicates on solutions.
        """
        if not self.config.multi_obj:
            raise ValueError("get_pf_solutions is only available for multi-objective optimization.")
        pf_solutions, _pf_fitness = self.get_pf(deduplicate)
        return pf_solutions

    def get_pf(self, deduplicate: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Approximate Pareto-front (solutions, un-negated fitness) over all evaluations.

        Requires ``full_fit_history`` and ``full_sol_history``; deduplicates on solutions.
        """
        if not self.config.multi_obj:
            raise ValueError("get_pf is only available for multi-objective optimization.")
        if not self.config.full_fit_history or not self.config.full_sol_history:
            warnings.warn("`get_pf` requires enabling both `full_sol_history` and `full_sol_history`.")
        all_solutions = np.concatenate(self.solution_history, axis=0)
        all_fitness = np.concatenate(self.fitness_history, axis=0)
        if deduplicate:
            _, unique_index = np.unique(all_solutions, axis=0, return_index=True)
            all_solutions = all_solutions[unique_index]
            all_fitness = all_fitness[unique_index]
        rank = self._non_dominate_rank(all_fitness)
        pf_mask = rank == 0
        return all_solutions[pf_mask], self._unn(all_fitness[pf_mask])

    # ------------------------------------------------------------ history

    def get_fitness_history(self) -> list[np.ndarray]:
        """Get the full history of fitness values (un-negated)."""
        return [self._unn(np.asarray(f)) for f in self.fitness_history]

    def get_solution_history(self) -> list[np.ndarray]:
        """Get the full history of solutions."""
        return self.solution_history

    def record_auxiliary(self, aux: dict) -> None:
        """Record one generation of algorithm auxiliary tensors (host-side).

        Delegates to the module-level :func:`record_auxiliary`; gated on
        ``full_pop_history``.
        """
        record_auxiliary(self.config, aux)

    @property
    def fitness_history(self) -> list[np.ndarray]:
        """Alias of ``fit_history`` (torch parity)."""
        return self.config.fit_history

    @property
    def fit_history(self) -> list[np.ndarray]:
        """The recorded per-generation fitness values."""
        return self.config.fit_history

    @property
    def solution_history(self) -> list[np.ndarray]:
        """Alias of ``sol_history`` (torch parity)."""
        return self.config.sol_history

    @property
    def sol_history(self) -> list[np.ndarray]:
        """The recorded per-generation solutions."""
        return self.config.sol_history

    @property
    def pop_history(self) -> list[np.ndarray]:
        """The recorded per-generation populations.

        Prefers the auxiliary ``"pop"`` channel when present (``record_auxiliary``
        with a ``"pop"`` key), otherwise falls back to the legacy
        ``config.pop_history`` list.
        """
        if "pop" in self.config.aux_history:
            return self.config.aux_history["pop"]
        return self.config.pop_history

    @property
    def aux_history(self) -> dict:
        """Algorithm auxiliary tensors, keyed by name, mapped to per-generation numpy arrays."""
        return self.config.aux_history

    @property
    def auxiliary_history(self) -> dict:
        """Alias of ``aux_history`` (torch parity)."""
        return self.config.aux_history

    # --------------------------------------------------------------- plot

    def plot(self, problem_pf: Any = None, source: str = "eval", **kwargs: Any) -> Any:
        """Plot the fitness history (or the algorithm's auxiliary "fit" channel).

        Host-side, torch-parity port of ``EvalMonitor.plot``: dispatches to the
        ``evox_etl.vis_tools`` figure builders (Plotly is an OPTIONAL dependency)
        by the number of objectives. Returns None — with a warning — when there is
        nothing to plot or when the visualization tool is unavailable.

        :param problem_pf: The problem's Pareto front to overlay (multi-objective
            only). Defaults to None.
        :param source: Which fitness to plot. ``"eval"`` (default) plots the
            problem-evaluation side, opt-direction un-negated via
            :meth:`get_fitness_history`; ``"pop"`` plots the RAW
            ``aux_history["fit"]`` channel the algorithm reported (NOT
            un-negated). Any other value raises ValueError.
        :param kwargs: Extra plot options forwarded to the ``vis_tools`` builder
            (e.g. ``animation=False`` for the 1-D figure).
        :return: A ``plotly.graph_objects.Figure``, or None when nothing can be
            plotted.
        """
        if not self.fitness_history and not self.aux_history:
            warnings.warn("No fitness history recorded, return None")
            return None

        vis_plot = _vis_plot_module()
        if vis_plot is None:
            warnings.warn(_NO_VIS_TOOL_WARNING)
            return None

        if source == "pop":
            fitness_history = self.aux_history["fit"]
        elif source == "eval":
            fitness_history = self.get_fitness_history()
        else:
            raise ValueError(f"Invalid source argument: {source}, expect 'eval' or 'pop'.")

        fitness_history = [np.asarray(f) for f in fitness_history]

        if fitness_history[0].ndim == 1:
            n_objs = 1
        else:
            n_objs = self.fitness_history[0].shape[1]

        if n_objs == 1:
            return vis_plot.plot_obj_space_1d(fitness_history, **kwargs)
        elif n_objs == 2:
            return vis_plot.plot_obj_space_2d(fitness_history, problem_pf, **kwargs)
        elif n_objs == 3:
            return vis_plot.plot_obj_space_3d(fitness_history, problem_pf, **kwargs)
        else:
            warnings.warn("Not supported yet.")
            return None


# The workflow resolves the wrapper class as ``Monitor`` in this module.
Monitor = EvalMonitor

__all__ = ["EvalMonitor"]


import warnings
import weakref
from enum import IntEnum
from typing import Dict, List, NamedTuple, Tuple

import torch

from evox.core import Monitor, Mutable
from evox.operators.selection import non_dominate_rank
from evox.utils import register_vmap_op

try:
    from evox.vis_tools import plot
except ImportError:
    plot = None


# https://github.com/pytorch/pytorch/issues/36748
def unique(x: torch.Tensor, dim=0):
    """Return the unique elements of the input tensor, as well as the unique index."""
    x = x.nan_to_num()
    unique, inverse, counts = torch.unique(x, dim=dim, sorted=True, return_inverse=True, return_counts=True)
    inv_sorted = inverse.argsort(stable=True)
    tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
    index = inv_sorted[tot_counts]
    return unique, inverse, counts, index


class HistoryType(IntEnum):
    """History type for the monitor."""

    FITNESS = 0
    SOLUTION = 1
    AUXILIARY = 2


class MonitorHistory(NamedTuple):
    fit_history: List[torch.Tensor]
    sol_history: List[torch.Tensor]
    aux_history: List[torch.Tensor]


__monitor_history__: Dict[int, MonitorHistory] = {}


def _fake_data_sink(monitor_id: int, data: torch.Tensor, data_type: int, token: torch.Tensor) -> torch.Tensor:
    return token.new_empty(token.size())


def _fake_vmap_data_sink(
    monitor_id: int,
    data: torch.Tensor,
    data_type: int,
    token: torch.Tensor,
) -> torch.Tensor:
    return token.new_empty(token.size())


def _vmap_data_sink(
    monitor_id: int,
    data: torch.Tensor,
    data_type: int,
    token: torch.Tensor,
) -> torch.Tensor:
    __monitor_history__[monitor_id][data_type].append(data)
    return token + 1


@register_vmap_op(fake_fn=_fake_data_sink, vmap_fn=_vmap_data_sink, fake_vmap_fn=_fake_vmap_data_sink)
def _data_sink(monitor_id: int, data: torch.Tensor, data_type: int, token: torch.Tensor) -> torch.Tensor:
    """Record the data into the monitor history log.

    This function uses the provided token to establish data dependencies between
    successive function calls, ensuring proper tracking and ordering of monitored values.
    """
    __monitor_history__[monitor_id][data_type].append(data)
    return token + 1


class EvalMonitor(Monitor):
    """Evaluation monitor.
    Used for both single-objective and multi-objective workflow.
    Hooked around the evaluation process,
    can monitor the offspring, their corresponding fitness and keep track of the evaluation count.
    Moreover, it can also record the best solution or the pareto front on-the-fly.
    """

    def __init__(
        self,
        multi_obj: bool = False,
        full_fit_history: bool = True,
        full_sol_history: bool = False,
        full_pop_history: bool = False,
        topk: int = 1,
        device: torch.device | None = None,
        history_device: torch.device | None = None,
    ):
        """Initialize the monitor.

        :param multi_obj: Whether the optimization is multi-objective. Defaults to False.
        :param full_fit_history: Whether to record the full history of fitness value. Default to True. Setting it to False may reduce memory usage.
        :param full_sol_history: Whether to record the full history of solutions. Default to False. Setting it to True may increase memory usage.
        :param topk: Only affect Single-objective optimization. The number of elite solutions to record. Default to 1, which will record the best individual.
        :param device: The device of the monitor. Defaults to None.
        :param history_device: The device to record the history. Defaults to None. If None, it will use cpu.

        ```{tip}
        Setting the `history_device` to the same device as the monitor will save the data transfer time,
        but may increase the memory usage on the device.
        ```

        ```{note}
        When `opt_direction="max"` is used,
        fitness values are internally multiplied by -1 to ensure that optimization logic always treats the best fitness as the minimum value.
        As a result, raw fitness values (e.g., `monitor.topk_fitness`, `monitor.fitness_history`, etc.) will appear negated.
        However, access methods such as `monitor.get_best_fitness()` and `monitor.get_pf_fitness()` automatically reverse this negation and return the original, unmodified values.
        ```
        """
        super().__init__()
        device = torch.get_default_device() if device is None else device
        history_device = torch.device("cpu") if history_device is None else history_device
        self.multi_obj = multi_obj
        self.full_fit_history = full_fit_history
        self.full_sol_history = full_sol_history
        self.full_pop_history = full_pop_history
        self.opt_direction = 1
        self.topk = topk
        self.device = device
        self.history_device = history_device
        self.aux_keys = []
        # mutable
        self.latest_solution = Mutable(torch.empty(0, device=device))
        self.latest_fitness = Mutable(torch.empty(0, device=device))
        self.topk_solutions = Mutable(torch.empty(0, device=device))
        self.topk_fitness = Mutable(torch.empty(0, device=device))
        self._id_ = id(self)
        self._token = Mutable(torch.tensor(0, device=device))
        __monitor_history__[self._id_] = MonitorHistory([], [], [])
        weakref.finalize(
            self,
            __monitor_history__.pop,
            self._id_,
            None,
        )

    @property
    def fitness_history(self) -> List[torch.Tensor]:
        return __monitor_history__[self._id_][HistoryType.FITNESS]

    @property
    def fit_history(self) -> List[torch.Tensor]:
        # alias for fitness_history
        return self.fitness_history

    @property
    def solution_history(self) -> List[torch.Tensor]:
        return __monitor_history__[self._id_][HistoryType.SOLUTION]

    @property
    def sol_history(self) -> List[torch.Tensor]:
        # alias for solution_history
        return self.solution_history

    @property
    def aux_history(self) -> Dict[str, List[torch.Tensor]]:
        # alias for auxiliary_history
        return self.auxiliary_history

    @property
    def auxiliary_history(self) -> Dict[str, List[torch.Tensor]]:
        raw_aux_history = __monitor_history__[self._id_][HistoryType.AUXILIARY]
        n_keys = len(self.aux_keys)
        if n_keys == 0:
            return {}

        assert len(raw_aux_history) % n_keys == 0
        aux_history = {}
        for i, key in enumerate(self.aux_keys):
            aux_history[key] = raw_aux_history[i::n_keys]

        return aux_history

    def set_config(self, **config):
        if "multi_obj" in config:
            self.multi_obj = config["multi_obj"]
        if "full_fit_history" in config:
            self.full_fit_history = config["full_fit_history"]
        if "full_sol_history" in config:
            self.full_sol_history = config["full_sol_history"]
        if "topk" in config:
            self.topk = config["topk"]
        if "opt_direction" in config:
            self.opt_direction = config["opt_direction"]
        return self

    def record_auxiliary(self, aux: Dict[str, torch.Tensor]):
        if self.full_pop_history:
            if len(self.aux_keys) == 0:
                self.aux_keys = list(aux.keys())

            for key in self.aux_keys:
                assert isinstance(aux[key], torch.Tensor)
                self._token = _data_sink(
                    self._id_, aux[key].to(self.history_device, non_blocking=True), HistoryType.AUXILIARY, self._token
                )

    def post_ask(self, candidate_solution: torch.Tensor):
        self.latest_solution = candidate_solution

    def pre_tell(self, fitness: torch.Tensor):
        self.latest_fitness = fitness
        if fitness.ndim == 1:
            # single-objective
            self.multi_obj = False
            assert fitness.size(0) >= self.topk
            if self.topk_solutions.ndim <= 1:
                topk_solutions = self.latest_solution
                topk_fitness = fitness
                rank = torch.topk(topk_fitness, self.topk, largest=False)[1]
                self.topk_fitness = topk_fitness[rank]
                self.topk_solutions = topk_solutions[rank]
            else:
                topk_solutions = torch.concatenate([self.topk_solutions, self.latest_solution])
                topk_fitness = torch.concatenate([self.topk_fitness, fitness])
                rank = torch.topk(topk_fitness, self.topk, largest=False)[1]
                self.topk_fitness = topk_fitness[rank]
                self.topk_solutions = topk_solutions[rank]
        elif fitness.ndim == 2:
            # multi-objective
            self.multi_obj = True
            # In multi-objective, we can't simply take the topk solutions.
            # Instead, we need to record the solutions and fitness values.
            # And in the end, we can get the pareto front.
        else:
            raise ValueError(f"Invalid fitness shape: {fitness.shape}")

        if self.full_fit_history or self.full_sol_history:
            self.record_history()

    def record_history(self):
        if self.full_sol_history:
            latest_solution = self.latest_solution.to(self.history_device, non_blocking=True)
            assert isinstance(latest_solution, torch.Tensor)
            self._token = _data_sink(self._id_, latest_solution, HistoryType.SOLUTION, self._token)
        if self.full_fit_history:
            latest_fitness = self.latest_fitness.to(self.history_device, non_blocking=True)
            assert isinstance(latest_fitness, torch.Tensor)
            self._token = _data_sink(self._id_, latest_fitness, HistoryType.FITNESS, self._token)

    def get_latest_fitness(self) -> torch.Tensor:
        """Get the fitness values from the latest iteration."""
        return self.opt_direction * self.latest_fitness

    def get_latest_solution(self) -> torch.Tensor:
        """Get the solution from the latest iteration."""
        return self.latest_solution

    def get_topk_fitness(self) -> torch.Tensor:
        """Get the topk fitness values so far."""
        return self.opt_direction * self.topk_fitness

    def get_topk_solutions(self) -> torch.Tensor:
        """Get the topk solutions so far."""
        if self.multi_obj:
            raise ValueError("Multi-objective optimization does not have a single best solution. Please use get_pf_solutions")
        return self.topk_solutions

    def get_best_solution(self) -> torch.Tensor:
        """Get the best solution so far."""
        if self.multi_obj:
            raise ValueError("Multi-objective optimization does not have a single best solution. Please use get_pf_solutions")
        return self.topk_solutions[0]

    def get_best_fitness(self) -> torch.Tensor:
        """Get the best fitness value so far."""
        if self.multi_obj:
            raise ValueError("Multi-objective optimization does not have a single best fitness. Please use get_pf_fitness")
        return self.opt_direction * self.topk_fitness[0]

    def get_pf_fitness(self, deduplicate=True) -> torch.Tensor:
        """Get the approximate pareto front fitness values of all the solutions evaluated so far.
        Requires enabling `full_fit_history`."""
        if not self.multi_obj:
            raise ValueError("get_pf_fitness is only available for multi-objective optimization.")
        if not self.full_fit_history:
            warnings.warn("`get_pf_fitness` requires enabling `full_fit_history`.")
        all_fitness = self.fitness_history
        all_fitness = torch.cat(all_fitness, dim=0)
        if deduplicate:
            all_fitness = torch.unique(all_fitness, dim=0)
        rank = non_dominate_rank(all_fitness)
        pf_fit = all_fitness[rank == 0]
        return pf_fit * self.opt_direction

    def get_pf_solutions(self, deduplicate=True) -> torch.Tensor:
        """Get the approximate pareto front solutions of all the solutions evaluated so far.
        Requires enabling both `full_sol_history` and `full_sol_history`.
        If `deduplicate` is set to True, the duplicated solutions will be removed."""
        if not self.multi_obj:
            raise ValueError("get_pf_solutions is only available for multi-objective optimization.")
        pf_solutions, _pf_fitness = self.get_pf(deduplicate)
        return pf_solutions

    def get_pf(self, deduplicate=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the approximate pareto front solutions and fitness values of all the solutions evaluated so far.
        Requires enabling both `full_sol_history` and `full_sol_history`.
        If `deduplicate` is set to True, the duplicated solutions will be removed."""
        if not self.multi_obj:
            raise ValueError("get_pf is only available for multi-objective optimization.")
        if not self.full_fit_history or not self.full_sol_history:
            warnings.warn("`get_pf` requires enabling both `full_sol_history` and `full_sol_history`.")
        all_solutions = self.get_solution_history()
        all_solutions = torch.cat(all_solutions, dim=0)
        all_fitness = self.fitness_history
        all_fitness = torch.cat(all_fitness, dim=0)

        if deduplicate:
            _, unique_index, _, _ = unique(all_solutions)
            all_solutions = all_solutions[unique_index]
            all_fitness = all_fitness[unique_index]

        rank = non_dominate_rank(all_fitness)
        pf_fitness = all_fitness[rank == 0]
        pf_solutions = all_solutions[rank == 0]
        return pf_solutions, pf_fitness * self.opt_direction

    def get_fitness_history(self) -> List[torch.Tensor]:
        """Get the full history of fitness values."""
        return [self.opt_direction * fit for fit in self.fitness_history]

    def get_solution_history(self) -> List[torch.Tensor]:
        """Get the full history of solutions."""
        return self.solution_history

    @torch.compiler.disable
    def plot(self, problem_pf=None, source="eval", **kwargs):
        """Plot the fitness history.
        If the problem's Pareto front is provided, it will be plotted as well.

        :param problem_pf: The Pareto front of the problem. Default to None.
        :param source: The source of the data, either "eval" or "pop", default to "eval".
            When "eval", the fitness from the problem evaluation side will be plotted, representing what the problem sees.
            When "pop", the fitness from the population inside the algorithm will be plotted, representing what the algorithm sees.
        :param kwargs: Additional arguments for the plot.
        """
        if not self.fitness_history and not self.aux_history:
            warnings.warn("No fitness history recorded, return None")
            return

        if plot is None:
            warnings.warn('No visualization tool available, return None. Hint: pip install "evox[vis]"')
            return

        if source == "pop":
            fitness_history = self.aux_history["fit"]
        elif source == "eval":
            fitness_history = self.get_fitness_history()
        else:
            raise ValueError(f"Invalid source argument: {source}, expect 'eval' or 'pop'.")

        fitness_history = [f.cpu().numpy() for f in fitness_history]

        if fitness_history[0].ndim == 1:
            n_objs = 1
        else:
            n_objs = self.fitness_history[0].shape[1]

        if n_objs == 1:
            return plot.plot_obj_space_1d(fitness_history, **kwargs)
        elif n_objs == 2:
            return plot.plot_obj_space_2d(fitness_history, problem_pf, **kwargs)
        elif n_objs == 3:
            return plot.plot_obj_space_3d(fitness_history, problem_pf, **kwargs)
        else:
            warnings.warn("Not supported yet.")

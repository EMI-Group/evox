import warnings
from typing import Callable, Dict, List, Optional, Union

from evox import Algorithm, Monitor, Problem, State, Stateful
from evox.utils import algorithm_has_init_ask, parse_opt_direction


class NonJitWorkflow(Stateful):
    def __init__(
        self,
        algorithm: Algorithm,
        problem: Problem,
        monitors: List[Monitor] = [],
        opt_direction: Union[str, List[str]] = "min",
        sol_transforms: List[Callable] = [],
        fit_transforms: List[Callable] = [],
        pop_transform: Optional[Callable] = None,
        monitor=None,
    ):
        """
        Parameters
        ----------
        algorithm
            The algorithm.
        problem
            The problem.
        monitor
            Optional monitor.
        opt_direction
            The optimization direction, can be either "min" or "max"
            or a list of "min"/"max" to specific the direction for each objective.
        sol_transforms
            Optional candidate solution transform function,
            usually used to decode the candidate solution
            into the format that can be understood by the problem.
            Should be a list of functions,
            and the functions will be applied in the order of the list.
        fit_transforms
            Optional fitness transform function.
            usually used to apply fitness shaping.
            Should be a list of functions,
            and the functions will be applied in the order of the list.
        """
        self.algorithm = algorithm
        self.problem = problem
        self.sol_transforms = sol_transforms
        if pop_transform is not None:
            warnings.warn(
                "`pop_transform` is deprecated, use `sol_transforms` with a list of transforms instead",
                DeprecationWarning,
            )
            self.sol_transforms = [pop_transform]
        self.fit_transforms = fit_transforms

        self.registered_hooks = {
            "pre_step": [],
            "pre_ask": [],
            "post_ask": [],
            "pre_eval": [],
            "post_eval": [],
            "pre_tell": [],
            "post_tell": [],
            "post_step": [],
        }
        self.monitors = monitors
        if monitor is not None:
            warnings.warn(
                "`monitor` is deprecated, use the `monitors` parameter with a list of monitors instead",
                DeprecationWarning,
            )
            self.monitors = [monitor]
        for monitor in self.monitors:
            hooks = monitor.hooks()
            for hook in hooks:
                self.registered_hooks[hook].append(monitor)

        self.opt_direction = parse_opt_direction(opt_direction)
        for monitor in self.monitors:
            monitor.set_opt_direction(self.opt_direction)

    def setup(self, key):
        return State(generation=0)

    def step(self, state):
        for monitor in self.registered_hooks["pre_step"]:
            monitor.pre_step(state)

        for monitor in self.registered_hooks["pre_ask"]:
            monitor.pre_ask(state)

        is_init = False
        if state.generation == 0:
            is_init = algorithm_has_init_ask(self.algorithm, state)
        else:
            is_init = False

        if is_init:
            cand_sol, state = self.algorithm.init_ask(state)
        else:
            cand_sol, state = self.algorithm.ask(state)

        for monitor in self.registered_hooks["post_ask"]:
            monitor.post_ask(state, cand_sol)

        transformed_cand_sol = cand_sol
        for transform in self.sol_transforms:
            transformed_cand_sol = transform(transformed_cand_sol)

        for monitor in self.registered_hooks["pre_eval"]:
            monitor.pre_eval(state, cand_sol, transformed_cand_sol)

        fitness, state = self.problem.evaluate(state, transformed_cand_sol)

        fitness = fitness * self.opt_direction

        for monitor in self.registered_hooks["post_eval"]:
            monitor.post_eval(state, cand_sol, transformed_cand_sol, fitness)

        transformed_fitness = fitness
        for transform in self.fit_transforms:
            transformed_fitness = transform(transformed_fitness)

        for monitor in self.registered_hooks["pre_tell"]:
            monitor.pre_tell(
                state, cand_sol, transformed_cand_sol, fitness, transformed_fitness
            )

        if is_init:
            state = self.algorithm.init_tell(state, fitness)
        else:
            state = self.algorithm.tell(state, fitness)

        for monitor in self.registered_hooks["post_tell"]:
            monitor.post_tell(state)

        for monitor in self.registered_hooks["post_step"]:
            monitor.post_step(state)

        return state.update(generation=state.generation + 1)

    def valid(self, state, metric="loss"):
        new_state = self.problem.valid(state, metric=metric)
        pop, new_state = self.algorithm.ask(new_state)
        new_state, fitness = self.problem.evaluate(new_state, pop)
        return fitness, state

    def sample(self, state):
        """Sample the algorithm but don't change it's state"""
        sample_pop, state_ = self.algorithm.ask(state)
        return sample_pop, state

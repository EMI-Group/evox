import warnings
from functools import partial
from typing import Callable, Dict, List, Optional, Union

import jax
import jax.numpy as jnp
from jax import jit, lax

from evox import Algorithm, Problem, State, Stateful, Monitor, jit_method
from evox.utils import parse_opt_direction, algorithm_has_init_ask


class MultitaskWorkflow(Stateful):
    """Multitask Workflow, designed to handle multitask optimization.
    Unlike StandardWorkflow, MultitaskWorkflow can handle multiple problems,
    meaning the `problems` parameter is be a list of problems.
    Likewise `opt_directions`, `sol_transforms`, and `fit_transforms` should be nested lists (2d lists) when specificed.
    Monitor should be specialized monitor for multitask optimization. For example `MultitaskEvalMonitor`
    """

    def __init__(
        self,
        algorithm: Algorithm,
        problems: Union[Problem, List[Problem]],
        monitors: List[Monitor] = [],
        opt_directions: Union[str, List[str]] = "min",
        sol_transforms: List[Callable] = [],
        fit_transforms: List[Callable] = [],
    ):
        """
        Parameters
        ----------
        algorithm
            The algorithm.
        problem
            The problem.
        monitor
            Optional monitor(s).
            Configure a single monitor or a list of monitors.
            The monitors will be called in the order of the list.
        opt_directions
            The optimization direction, can be either "min" or "max"
            or a list of "min"/"max" to specific the direction for each objective.
        sol_transform
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
        jit_problem
            If the problem can be jit compiled by JAX or not.
            Default to True.
        num_objectives
            Number of objectives.
            When the problem can be jit compiled, this field is not needed.
            When the problem cannot be jit compiled, this field should be set,
            if not, default to 1.
        """
        self.algorithm = algorithm
        self.problems = problems
        self.monitors = monitors
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
        for monitor in self.monitors:
            hooks = monitor.hooks()
            for hook in hooks:
                self.registered_hooks[hook].append(monitor)

        if isinstance(opt_directions, str):
            opt_directions = [opt_directions] * len(self.problems)
        self.opt_directions = [parse_opt_direction(d) for d in opt_directions]
        for monitor in self.monitors:
            monitor.set_opt_direction(self.opt_directions)

        self.sol_transforms = sol_transforms
        # for compatibility purpose
        self.fit_transforms = fit_transforms

        # a prototype step function
        # will be then wrapped to get _step
        # We are doing this as a workaround for JAX's static shape requirement
        # Since init_ask and ask can return different shape
        # and jax.lax.cond requires the same shape from two different branches
        # we can only apply lax.cond outside of each `step`
        def _proto_step(self, is_init, state):
            for monitor in self.registered_hooks["pre_ask"]:
                monitor.pre_ask(state)

            if is_init:
                ask = self.algorithm.init_ask
                tell = self.algorithm.init_tell
            else:
                ask = self.algorithm.ask
                tell = self.algorithm.tell

            # candidate solution
            cand_sols, state = ask(state)
            num_tasks = len(cand_sols)

            for monitor in self.registered_hooks["post_ask"]:
                monitor.post_ask(state, cand_sols)

            for monitor in self.registered_hooks["pre_eval"]:
                monitor.pre_eval(state, cand_sols, cand_sols)

            # if the function is jitted
            all_task_fitness = []
            for opt_dir, problem, cand_sol in zip(
                self.opt_directions, self.problems, cand_sols
            ):
                fitness, state = problem.evaluate(state, cand_sol)
                fitness = fitness * opt_dir
                all_task_fitness.append(fitness)

            for monitor in self.registered_hooks["post_eval"]:
                monitor.post_eval(state, cand_sols, cand_sols, all_task_fitness)

            for monitor in self.registered_hooks["pre_tell"]:
                monitor.pre_tell(
                    state, cand_sols, cand_sols, all_task_fitness, all_task_fitness
                )

            state = tell(state, all_task_fitness)

            for monitor in self.registered_hooks["post_tell"]:
                monitor.post_tell(state)

            return state.update(generation=state.generation + 1)

        # wrap around _proto_step
        # to handle init_ask and init_tell
        def _step(self, state):
            # probe if self.algorithm has override the init_ask function
            if algorithm_has_init_ask(self.algorithm, state):
                return lax.cond(
                    state.generation == 0,
                    partial(_proto_step, self, True),
                    partial(_proto_step, self, False),
                    state,
                )
            else:
                return _proto_step(self, False, state)

        # the first argument is self, which should be static
        self._step = jit(_step, static_argnums=[0])

    def setup(self, _key):
        return State(generation=0)

    def step(self, state):
        for monitor in self.registered_hooks["pre_step"]:
            monitor.pre_step(state)

        state = self._step(self, state)

        for monitor in self.registered_hooks["post_step"]:
            monitor.post_step(state)

        return state

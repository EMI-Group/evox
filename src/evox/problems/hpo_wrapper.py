import weakref
from abc import ABC
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import torch
from torch import nn

from evox.core import Monitor, Mutable, Problem, Workflow, compile, use_state, vmap


def _vmap_vmap_mean_fit_aggregation(info, in_dims, fit: torch.Tensor) -> Tuple[torch.Tensor, int]:
    return torch.mean(fit.movedim(in_dims[0], 0), dim=0, keepdim=True), 0


@torch.library.custom_op("evox::_hpo_vmap_mean_fit_aggregation", mutates_args=())
def _vmap_mean_fit_aggregation(fit: torch.Tensor) -> torch.Tensor:
    return fit.clone()


_vmap_mean_fit_aggregation.register_fake(lambda f: f.new_empty(f.size()))
_vmap_mean_fit_aggregation.register_vmap(_vmap_vmap_mean_fit_aggregation)


@torch.library.custom_op("evox::_hpo_mean_fit_aggregation", mutates_args=())
def _mean_fit_aggregation(fit: torch.Tensor) -> torch.Tensor:
    return fit.clone()


_mean_fit_aggregation.register_fake(lambda f: f.new_empty(f.size()))
_mean_fit_aggregation.register_vmap(lambda info, in_dims, fit: (_vmap_mean_fit_aggregation(fit.movedim(in_dims[0], 0)), 0))


class HPOMonitor(Monitor, ABC):
    """The base class for hyper parameter optimization (HPO) monitors used in `HPOProblem.workflow.monitor`."""

    def __init__(
        self,
        num_repeats: int = 1,
        fit_aggregation: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = _mean_fit_aggregation,
    ):
        super().__init__()
        self.num_repeats = num_repeats
        self.fit_aggregation = fit_aggregation

    def tell_fitness(self) -> torch.Tensor:
        """Get the best fitness found so far in the optimization process that this monitor is monitoring.

        :return: The best fitness so far.
        """
        raise NotImplementedError("`tell_fitness` function is not implemented. It must be overwritten.")


class HPOFitnessMonitor(HPOMonitor):
    """The monitor for hyper parameter optimization (HPO) that records the best fitness found so far in the optimization process."""

    def __init__(
        self,
        num_repeats: int = 1,
        fit_aggregation: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = _mean_fit_aggregation,
        multi_obj_metric: Optional[Callable] = None,
    ):
        """
        Initialize the HPO fitness monitor.

        :param multi_obj_metric: The metric function to use for multi-objective optimization, unused in single-objective optimization.
            Currently we only support "IGD" or "HV" for multi-objective optimization. Defaults to `None`.
        """
        super().__init__(num_repeats, fit_aggregation)
        assert multi_obj_metric is None or callable(multi_obj_metric), (
            f"Expect `multi_obj_metric` to be `None` or callable, got {multi_obj_metric}"
        )
        self.multi_obj_metric = multi_obj_metric
        self.best_fitness = Mutable(torch.tensor(torch.inf))

    def pre_tell(self, fitness: torch.Tensor):
        """Update the best fitness value found so far based on the provided fitness tensor and multi-objective metric.

        :param fitness: A tensor representing fitness values. It can be either a 1D tensor for single-objective optimization or a 2D tensor for multi-objective optimization.

        :raises AssertionError: If the dimensionality of the fitness tensor is not 1 or 2.
        """
        fitness = self.fit_aggregation(fitness) if self.num_repeats > 1 else fitness
        if fitness.ndim == 1:
            # single-objective
            self.best_fitness = torch.min(torch.min(fitness), self.best_fitness)
        else:
            # multi-objective
            self.best_fitness = torch.min(self.multi_obj_metric(fitness), self.best_fitness)

    def tell_fitness(self) -> torch.Tensor:
        """Get the best fitness found so far in the optimization process that this monitor is monitoring.

        :return: The best fitness so far.
        """
        return self.best_fitness


def get_sub_state(state: Dict[str, Any], name: str):
    """Get the sub state from the tuple of states.

    :param state: The tuple of states.

    :return: The sub state.
    """
    prefix_len = len(name) + 1
    state = {k[prefix_len:]: v for k, v in state.items() if k.startswith(name)}
    return state


class HPOData(NamedTuple):
    workflow_step: Callable[[Dict[str, torch.Tensor]], Tuple[Dict[str, torch.Tensor]]]  # workflow_step
    compiled_workflow_step: Callable[[Dict[str, torch.Tensor]], Tuple[Dict[str, torch.Tensor]]]  # compiled_workflow_step
    state_keys: List[str]  # state_keys or param_keys
    buffer_keys: Optional[List[str]]  # optional buffer_keys


__hpo_data__: Dict[int, HPOData] = {}


def _fake_hpo_evaluate_loop(id: int, iterations: int, state_values: List[torch.Tensor]) -> List[torch.Tensor]:
    return [v.new_empty(v.size()) for v in state_values]


@torch.library.custom_op("evox::_hpo_evaluate_loop", mutates_args=())
def _hpo_evaluate_loop(id: int, iterations: int, state_values: List[torch.Tensor]) -> List[torch.Tensor]:
    global __hpo_data__
    workflow_step, compiled_workflow_step, state_keys, buffer_keys = __hpo_data__[id]
    if buffer_keys is None:
        state = {k: v.clone() for k, v in zip(state_keys, state_values)}
        for _ in range(iterations):
            if torch.compiler.is_compiling():
                state = compiled_workflow_step(state)
            else:
                state = workflow_step(state)
        return [state[k] for k in state_keys]
    else:
        param_keys, buffer_keys = state_keys, buffer_keys
        params = {k: v.clone() for k, v in zip(param_keys, state_values)}
        buffers = {k: v.clone() for k, v in zip(buffer_keys, state_values[len(param_keys) :])}
        for _ in range(iterations):
            if torch.compiler.is_compiling():
                params, buffers = compiled_workflow_step(params, buffers)
            else:
                params, buffers = workflow_step(params, buffers)
        return [params[k] for k in param_keys] + [buffers[k] for k in buffer_keys]


_hpo_evaluate_loop.register_fake(_fake_hpo_evaluate_loop)


class HPOProblemWrapper(Problem):
    """The problem for hyper parameter optimization (HPO).

    ## Example
    ```python
    algo = SomeAlgorithm(...)
    prob = SomeProblem(...)
    monitor = HPOFitnessMonitor()
    workflow = StdWorkflow(algo, prob, monitor=monitor)
    hpo_prob = HPOProblemWrapper(iterations=..., num_instances=...)
    params = hpo_prob.get_init_params()
    # alter `params` ...
    hpo_prob.evaluate(params) # execute the evaluation
    # ...
    ```
    """

    def __init__(
        self,
        iterations: int,
        num_instances: int,
        workflow: Workflow,
        num_repeats: int = 1,
        copy_init_state: bool = False,
    ):
        """Initialize the HPO problem wrapper.

        :param iterations: The number of iterations to be executed in the optimization process.
        :param num_instances: The number of instances to be executed in parallel in the optimization process, i.e., the population size of the outer algorithm.
        :param workflow: The workflow to be used in the optimization process. Must be wrapped by `core.jit_class`.
        :param num_repeats: The number of times to repeat the evaluation process for each instance. Defaults to 1.
        :param copy_init_state: Whether to copy the initial state of the workflow for each evaluation. Defaults to `True`. If your workflow contains operations that IN-PLACE modify the tensor(s) in initial state, this should be set to `True`. Otherwise, you can set it to `False` to save memory.
        """
        super().__init__()
        assert iterations > 0, f"`iterations` should be greater than 0, got {iterations}"
        assert num_instances > 0, f"`num_instances` should be greater than 0, got {num_instances}"
        self.iterations = iterations
        self.num_instances = num_instances
        self.num_repeats = num_repeats
        self.copy_init_state = copy_init_state
        # check monitor
        monitor = workflow.monitor
        assert isinstance(monitor, HPOMonitor), f"Expect workflow monitor to be `HPOMonitor`, got {type(monitor)}"
        monitor.num_repeats = num_repeats

        # compile workflow steps
        state_step = use_state(workflow.step)

        def repeat_state_step(params: Dict[str, torch.Tensor], buffers: Dict[str, torch.Tensor]):
            state = {**params, **buffers}
            state = state_step(state)
            return {k: state[k] for k in params.keys()}, {k: state[k] for k in buffers.keys()}

        vmap_state_step = (
            torch.vmap(
                torch.vmap(repeat_state_step, randomness="same"),
                randomness="different",
                in_dims=(None, 0),
                out_dims=(None, 0),
            )
            if num_repeats > 1
            else torch.vmap(state_step, randomness="same")
        )
        self._init_params, self._init_buffers = torch.func.stack_module_state([workflow] * self.num_instances)
        if num_repeats > 1:
            self._init_buffers = {k: torch.stack([v] * num_repeats) for k, v in self._init_buffers.items()}
        self._workflow_step_ = vmap_state_step
        self._compiled_workflow_step_ = compile(vmap_state_step, fullgraph=True)

        if type(workflow).init_step == Workflow.init_step:
            # if no init step
            self._workflow_init_step_ = self._workflow_step_
            self._compiled_init_step_ = self._compiled_workflow_step_
        else:
            # otherwise, compile workflow init step
            state_init_step = use_state(workflow.init_step)

            def repeat_state_init_step(params: Dict[str, torch.Tensor], buffers: Dict[str, torch.Tensor]):
                state = {**params, **buffers}
                state = state_step(state)
                return {k: state[k] for k in params.keys()}, {k: state[k] for k in buffers.keys()}

            vmap_state_init_step = (
                torch.vmap(
                    torch.vmap(repeat_state_init_step, randomness="same"),
                    randomness="different",
                    in_dims=(None, 0),
                    out_dims=(None, 0),
                )
                if num_repeats > 1
                else torch.vmap(state_init_step, randomness="same")
            )
            self._workflow_init_step_ = vmap_state_init_step
            self._compiled_workflow_init_step_ = compile(vmap_state_init_step, fullgraph=True)

        self.state_keys = (list(self._init_params.keys()), list(self._init_buffers.keys()))
        if self.num_repeats == 1:
            self.state_keys = sum(self.state_keys, [])
        global __hpo_data__
        __hpo_data__[id(self)] = HPOData(
            workflow_step=self._workflow_step_,
            compiled_workflow_step=self._compiled_workflow_step_,
            state_keys=self.state_keys if self.num_repeats == 1 else self.state_keys[0],
            buffer_keys=None if self.num_repeats == 1 else self.state_keys[1],
        )
        self._id_ = id(self)
        weakref.finalize(self, __hpo_data__.pop, id(self), None)

        self._stateful_tell_fitness = use_state(monitor.tell_fitness)

    def evaluate(self, hyper_parameters: Dict[str, nn.Parameter]):
        """
        Evaluate the fitness (given by the internal workflow's monitor) of the batch of hyper parameters by running the internal workflow.

        :param hyper_parameters: The hyper parameters to evaluate.

        :return: The final fitness of the hyper parameters.
        """
        # hyper parameters check
        for k, _ in hyper_parameters.items():
            assert k in self._init_params, (
                f"`{k}` should be a hyperparameter of the workflow, available keys are {self.get_params_keys()}"
            )

        if self.num_repeats > 1:
            if self.copy_init_state:
                params = {k: v.clone() for k, v in self._init_params.items()}
                buffers = {k: v.clone() for k, v in self._init_buffers.items()}
            else:
                params = self._init_params
                buffers = self._init_buffers
            params = {**self._init_params, **hyper_parameters}
            # run the workflow
            if torch.compiler.is_compiling():
                params, buffers = self._compiled_workflow_init_step_(params, buffers)
            else:
                params, buffers = self._workflow_init_step_(params, buffers)
            state_values = [params[k] for k in self.state_keys[0]] + [buffers[k] for k in self.state_keys[1]]
            state_values = _hpo_evaluate_loop(self._id_, self.iterations - 1, state_values)
            params = {k: v for k, v in zip(self.state_keys[0], state_values)}
            buffers = {k: v for k, v in zip(self.state_keys[1], state_values[len(params) :])}
            monitor_state = get_sub_state(buffers, "monitor")
            _, fit = vmap(torch.vmap(self._stateful_tell_fitness))(monitor_state)
            return fit[0]
        else:
            state: Dict[str, torch.Tensor] = {**self._init_params, **self._init_buffers}
            if self.copy_init_state:
                state = {k: v.clone() for k, v in state.items()}
            # Override with the given hyper parameters
            state.update(hyper_parameters)
            # run the workflow
            if torch.compiler.is_compiling():
                state = self._compiled_workflow_init_step_(state)
            else:
                state = self._workflow_init_step_(state)
            state_values = [state[k] for k in self.state_keys]
            state_values = _hpo_evaluate_loop(self._id_, self.iterations - 1, state_values)
            state = {k: v for k, v in zip(self.state_keys, state_values)}
            monitor_state = get_sub_state(state, "monitor")
            _, fit = vmap(self._stateful_tell_fitness)(monitor_state)
            return fit

    def get_init_params(self):
        """Return the initial hyper-parameters dictionary of the underlying workflow."""
        return self._init_params

    def get_params_keys(self):
        return list(self._init_params.keys())

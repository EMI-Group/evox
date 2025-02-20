from abc import ABC
from typing import Callable, Dict, Optional, Tuple, Any
import copy

import torch
from torch import nn

from ..core import Monitor, Mutable, Problem, Workflow, use_state, vmap


class HPOMonitor(Monitor, ABC):
    """The base class for hyper parameter optimization (HPO) monitors used in `HPOProblem.workflow.monitor`."""

    def __init__(self):
        super().__init__()

    def tell_fitness(self) -> torch.Tensor:
        """Get the best fitness found so far in the optimization process that this monitor is monitoring.

        :return: The best fitness so far.
        """
        raise NotImplementedError("`tell_fitness` function is not implemented. It must be overwritten.")


class HPOFitnessMonitor(HPOMonitor):
    """The monitor for hyper parameter optimization (HPO) that records the best fitness found so far in the optimization process."""

    def __init__(self, multi_obj_metric: Optional[Callable] = None):
        """
        Initialize the HPO fitness monitor.

        :param multi_obj_metric: The metric function to use for multi-objective optimization, unused in single-objective optimization.
            Currently we only support "IGD" or "HV" for multi-objective optimization. Defaults to `None`.
        """
        super().__init__()
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
        assert 1 <= fitness.ndim <= 2
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


def get_sub_state(state: Tuple[Dict[str, Any], Dict[str, Any]], name: str):
    """Get the sub state from the tuple of states.

    :param state: The tuple of states.

    :return: The sub state.
    """
    params, buffers = state
    prefix_len = len(name) + 1
    params = {k[prefix_len:]: v for k, v in params.items() if k.startswith(name)}
    buffers = {k[prefix_len:]: v for k, v in buffers.items() if k.startswith(name)}
    return params, buffers


class HPOProblemWrapper(Problem):
    """The problem for hyper parameter optimization (HPO).

    ## Usage
    ```
    algo = SomeAlgorithm(...)
    algo.setup(...)
    prob = SomeProblem(...)
    prob.setup(...)
    monitor = HPOFitnessMonitor()
    workflow = StdWorkflow()
    workflow.setup(algo, prob, monitor=monitor)
    hpo_prob = HPOProblemWrapper(iterations=..., num_instances=...)
    hpo_prob.setup(workflow)
    params = HPOProblemWrapper.extract_parameters(hpo_prob.init_state)
    hpo_prob.evaluate(params) # execute the evaluation
    # ...
    ```
    """

    def __init__(self, iterations: int, num_instances: int, workflow: Workflow, copy_init_state: bool = True):
        """Initialize the HPO problem wrapper.

        :param iterations: The number of iterations to be executed in the optimization process.
        :param num_instances: The number of instances to be executed in parallel in the optimization process.
        :param workflow: The workflow to be used in the optimization process. Must be wrapped by `core.jit_class`.
        :param copy_init_state: Whether to copy the initial state of the workflow for each evaluation. Defaults to `True`. If your workflow contains operations that IN-PLACE modify the tensor(s) in initial state, this should be set to `True`. Otherwise, you can set it to `False` to save memory.
        """
        super().__init__()
        assert iterations > 0, f"`iterations` should be greater than 0, got {iterations}"
        assert num_instances > 0, f"`num_instances` should be greater than 0, got {num_instances}"
        self.iterations = iterations
        self.num_instances = num_instances
        self.copy_init_state = copy_init_state
        # check monitor
        monitor = workflow.monitor
        assert isinstance(monitor, HPOMonitor), f"Expect workflow monitor to be `HPOMonitor`, got {type(monitor)}"
        self.hpo_monitor = monitor
        state_step = use_state(workflow.step)

        # JIT workflow step
        vmap_state_step = vmap(state_step, randomness="different")
        init_state = torch.func.stack_module_state([workflow] * self.num_instances)
        self._workflow_step_ = torch.compile(vmap_state_step)
        self._init_state_ = init_state
        # self._get_monitor_fitness_ = torch.compile(get_monitor_fitness)
        if type(workflow).init_step == Workflow.init_step:
            # if no init step
            self._workflow_init_step_ = self._workflow_step_
        else:
            # otherwise, JIT workflow init step
            state_init_step = use_state(workflow.init_step)
            vmap_state_init_step = vmap(state_init_step, randomness="different")
            self._workflow_init_step_ = torch.compile(vmap_state_init_step)

    def evaluate(self, hyper_parameters: Dict[str, nn.Parameter]):
        """
        Evaluate the fitness (given by the internal workflow's monitor) of the batch of hyper parameters by running the internal workflow.

        :param hyper_parameters: The hyper parameters to evaluate.

        :return: The final fitness of the hyper parameters.
        """
        if self.copy_init_state:
            state = copy.deepcopy(self._init_state_)
        else:
            state = self._init_state_
        params, buffers = self._init_state_
        # hyper parameters check
        for k, v in hyper_parameters.items():
            assert k in params, f"`{k}` should be a hyperparameter of the workflow, available keys are {list(params.keys())}"
        # run the workflow
        params.update(hyper_parameters)
        state = (params, buffers)
        state = self._workflow_init_step_(state)
        for _ in range(self.iterations - 1):
            state = self._workflow_step_(state)
        # get final fitness
        monitor_state = get_sub_state(state, "monitor")
        fit, _monitor_state = vmap(use_state(self.hpo_monitor.tell_fitness), randomness="same")(monitor_state)
        return fit

    @torch.jit.ignore
    def get_init_params(self):
        """Return the initial hyper-parameters dictionary of the underlying workflow."""
        init_params, _init_buffers = self._init_state_
        return init_params

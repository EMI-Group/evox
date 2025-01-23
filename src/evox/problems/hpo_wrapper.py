from typing import Callable, Dict, Optional

import torch
from torch import nn

from ..core import Monitor, Mutable, Problem, Workflow, jit, jit_class, use_state, vmap, vmap_impl
from ..core._vmap_fix import unwrap_batch_tensor, wrap_batch_tensor
from ..core.module import _WrapClassBase


class HPOMonitor(Monitor):
    """The base class for hyper parameter optimization (HPO) monitors used in `HPOProblem.workflow.monitor`."""

    def __init__(self, num_repeats: int = 1):
        """
        Initialize the HPO monitor.

        :param num_repeats: The number of workflow repeats to be executed in the optimization process. Defaults to 1.
        """
        super().__init__()
        self.num_repeats = num_repeats

    def vmap_fitness_unwrap_repeats(self, fitness: torch.Tensor):
        """
        Unwrap the `num_repeats` batch dimension from the given fitness tensor.

        :param fitness: The fitness tensor to be unwrapped.
        :return: The unwrapped fitness tensor.
        """
        assert 1 <= fitness.ndim <= 2
        original_fitness, batch_dims, _ = unwrap_batch_tensor(fitness)
        assert batch_dims == (0,)
        new_fitness = original_fitness.view(self.num_repeats, -1, *fitness.size())
        pop_size = new_fitness.size(1)
        new_fitness = wrap_batch_tensor(new_fitness, (1,))
        return new_fitness, pop_size

    def tell_fitness(self) -> torch.Tensor:
        """Get the best fitness found so far in the optimization process that this monitor is monitoring.

        :return: The best fitness so far.
        """
        raise NotImplementedError("`tell_fitness` function is not implemented. It must be overwritten.")


@jit_class
class HPOFitnessMonitor(HPOMonitor):
    """The monitor for hyper parameter optimization (HPO) that records the best fitness found so far in the optimization process."""

    def __init__(
        self,
        *,
        num_repeats: int = 1,
        fit_aggregation: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = torch.mean,
        multi_obj_metric: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        """
        Initialize the HPO fitness monitor.

        :param num_repeats: The number of workflow repeats to be executed in the optimization process. Defaults to 1.
        :param fit_aggregation: The aggregation function to use for fitness aggregation when `num_repeats` > 1. Defaults to `torch.mean`.
        :param multi_obj_metric: The metric function to use for multi-objective optimization, unused in single-objective optimization. Defaults to `None`.
        """
        super().__init__(num_repeats=num_repeats)
        assert multi_obj_metric is None or callable(multi_obj_metric), (
            f"Expect `multi_obj_metric` to be `None` or callable, got {multi_obj_metric}"
        )
        self.fit_aggregation = fit_aggregation
        self.multi_obj_metric = multi_obj_metric
        self.best_fitness = Mutable(torch.tensor(torch.inf))

    def _fitness_unwrap(self, fitness: torch.Tensor):
        return fitness

    @vmap_impl(_fitness_unwrap)
    def _vmap_fitness_unwrap(self, fitness: torch.Tensor):
        fitness, pop_size = self.vmap_fitness_unwrap_repeats(fitness)
        original_best = unwrap_batch_tensor(self.best_fitness)[0]
        if original_best.size(0) != pop_size:
            self.best_fitness = wrap_batch_tensor(original_best[:pop_size], 0)
        return fitness

    def _fitness_wrap(self, best_fitness: torch.Tensor):
        return best_fitness

    @vmap_impl(_fitness_wrap)
    def _vmap_fitness_wrap(self, best_fitness: torch.Tensor):
        original_best = unwrap_batch_tensor(best_fitness)[0]
        original_best = original_best.repeat(self.num_repeats)
        return wrap_batch_tensor(original_best, 0)

    def pre_tell(self, fitness: torch.Tensor):
        """Update the best fitness value found so far based on the provided fitness tensor and multi-objective metric.

        :param fitness: A tensor representing fitness values. It can be either a 1D tensor for single-objective optimization or a 2D tensor for multi-objective optimization.

        :raises AssertionError: If the dimensionality of the fitness tensor is not 1 or 2.
        """
        fitness = self._fitness_unwrap(fitness)
        has_repeat = 1 if self.num_repeats > 1 else 0
        assert 1 <= fitness.ndim - has_repeat <= 2
        if fitness.ndim - has_repeat == 1:
            # single-objective
            if self.num_repeats > 1:
                fitness = self.fit_aggregation(fitness, dim=0)
            best_fitness = torch.min(torch.min(fitness), self.best_fitness)
        else:
            # multi-objective
            if self.num_repeats > 1:
                fitness = self.fit_aggregation(fitness, dim=0)
            best_fitness = torch.min(self.multi_obj_metric(fitness), self.best_fitness)
        self.best_fitness = self._fitness_wrap(best_fitness)

    def tell_fitness(self) -> torch.Tensor:
        """Get the best fitness found so far in the optimization process that this monitor is monitoring.

        :return: The best fitness so far.
        """
        return self.best_fitness

    @vmap_impl(tell_fitness)
    def _vmap_tell_fitness(self) -> torch.Tensor:
        actual_best = unwrap_batch_tensor(self.best_fitness)[0].view(self.num_repeats, -1)[0]
        return wrap_batch_tensor(actual_best, 0)


@jit_class
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

    def __init__(
        self, iterations: int, num_instances: int, workflow: Workflow, num_repeats: int = 1, copy_init_state: bool = True
    ):
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
        self.num_repeats = num_repeats
        self.copy_init_state = copy_init_state
        # compile workflow steps
        assert isinstance(workflow, _WrapClassBase), f"Expect `workflow` to be wrapped by `jit_class`, got {type(workflow)}"
        workflow.__sync__()
        # check monitor
        monitor = workflow.get_submodule("monitor")
        assert isinstance(monitor, HPOMonitor), f"Expect workflow monitor to be `HPOMonitor`, got {type(monitor)}"
        monitor.num_repeats = num_repeats
        monitor_state = monitor.state_dict(keep_vars=True)
        state_step = use_state(lambda: workflow.step)
        # get monitor's corresponding keys in init_state
        non_batched_init_state = state_step.init_state(clone=False).items()
        hyper_param_keys = []
        monitor_keys = {}
        for k, v in non_batched_init_state:
            for sk, sv in monitor_state.items():
                if sv is v:
                    monitor_keys[k] = sk
                    break
            if isinstance(v, nn.Parameter):
                hyper_param_keys.append(k)
        self._hyper_param_keys_ = hyper_param_keys
        assert len(monitor_keys) == len(monitor_state), (
            f"Expect monitor to have {len(monitor_state)} parameters, got {len(monitor_keys)}"
        )

        def get_monitor_fitness(x: Dict[str, torch.Tensor]):
            final_monitor_state = {sk: x[k] for k, sk in monitor_keys.items()}
            monitor.load_state_dict(final_monitor_state)
            return monitor.tell_fitness()

        # JIT workflow step
        vmap_state_step = vmap(state_step)
        init_state = vmap_state_step.init_state(self.num_instances * self.num_repeats)
        self._workflow_step_: torch.jit.ScriptFunction = jit(vmap_state_step, trace=True, example_inputs=(init_state,))
        self._get_monitor_fitness_ = jit(vmap(get_monitor_fitness, trace=False), trace=True, example_inputs=(init_state,))
        monitor.load_state_dict(monitor_state)
        # if no init step
        if type(workflow).init_step == Workflow.init_step:
            self._init_state_ = init_state
            self._workflow_init_step_ = self._workflow_step_
            return
        # otherwise, JIT workflow init step
        state_init_step = use_state(lambda: workflow.init_step)
        vmap_state_init_step = vmap(state_init_step)
        self._init_state_ = vmap_state_init_step.init_state(self.num_instances * self.num_repeats)
        self._workflow_init_step_: torch.jit.ScriptFunction = jit(
            vmap_state_init_step, trace=True, example_inputs=(self._init_state_,)
        )

        # JIT expand hyperparameters
        def expand_hyper_params(hyper_parameters: Dict[str, torch.Tensor]):
            if num_repeats > 1:
                return {k: v.repeat(num_repeats, *([1] * (v.ndim - 1))) for k, v in hyper_parameters.items()}
            else:
                return hyper_parameters

        self._expand_hp_ = jit(
            expand_hyper_params,
            trace=True,
            example_inputs=({k: v for k, v in self._init_state_.items() if k in hyper_param_keys},),
        )

    def evaluate(self, hyper_parameters: Dict[str, nn.Parameter]):
        """
        Evaluate the fitness (given by the internal workflow's monitor) of the batch of hyper parameters by running the internal workflow.

        :param hyper_parameters: The hyper parameters to evaluate.

        :return: The final fitness of the hyper parameters.
        """
        # hyper parameters check
        for k, v in hyper_parameters.items():
            assert k in self._init_state_, f"`{k}` should be in state dict of workflow and is `torch.nn.Parameter`"
            assert isinstance(self._init_state_[k], nn.Parameter) and isinstance(v, nn.Parameter), (
                f"`{k}` should correspond to a `torch.nn.Parameter`, got {type(self._init_state_[k])} and {type(v)}"
            )
        # run the workflow
        state = {}
        if self.copy_init_state:
            for k, v in self._init_state_.items():
                state[k] = v.clone()
        else:
            state = self._init_state_
        hyper_parameters = self._expand_hp_(hyper_parameters)
        state.update(hyper_parameters)
        state = self._workflow_init_step_(state)
        for _ in range(self.iterations - 1):
            state = self._workflow_step_(state)
        # get final fitness
        return self._get_monitor_fitness_(state)

    @torch.jit.ignore
    def get_init_params(self):
        """Return the initial hyper-parameters dictionary of the underlying workflow."""
        return {k: v for k, v in self._init_state_.items() if k in self._hyper_param_keys_}

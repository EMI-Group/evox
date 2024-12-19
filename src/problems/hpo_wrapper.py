from typing import Dict, Optional

import torch
from torch import nn

from ..core import Problem, Workflow, Monitor, jit_class, use_state, vmap, jit
from .. import utils


class HpoFitnessMonitor(Monitor):
    def __init__(self, multi_obj_indicator: Optional[str] = None):
        super().__init__()
        assert multi_obj_indicator is None or isinstance(
            multi_obj_indicator, str
        ), f"Expect `multi_obj_indicator` to be `None` or `str`, got {multi_obj_indicator}"
        multi_obj_indicator = multi_obj_indicator.capitalize()
        assert multi_obj_indicator in [
            "IGD",
            "HV",
        ], f"Currently we only support `IGD` or `HV`, got {multi_obj_indicator}"
        self.multi_obj_indicator = multi_obj_indicator

    def setup(self):
        super().setup()
        self.best_fitness = nn.Buffer(torch.tensor(torch.inf))
        return self

    def pre_tell(self, fitness: torch.Tensor):
        if fitness.ndim == 1:
            # single-objective
            self.best_fitness = torch.min(torch.min(fitness), self.best_fitness)
        else:
            pass
            # # multi-objective, TODO: add indicators
            # if self.multi_obj_indicator == "IGD":
            #     self.best_fitness = torch.min(utils.IGD(fitness), self.best_fitness)
            # elif self.multi_obj_indicator == "HV":
            #     self.best_fitness = torch.max(utils.HV(fitness), self.best_fitness)


@jit_class
class HpoProblemWrapper(Problem):

    def __init__(self, iterations: int, num_instances: int):
        super().__init__()
        assert iterations > 0, f"`iterations` should be greater than 0, got {iterations}"
        assert num_instances > 0, f"`num_instances` should be greater than 0, got {num_instances}"
        self.iterations = iterations
        self.num_instances = num_instances

    def setup(self, workflow: Workflow):
        self.workflow = workflow
        assert "monitor" in workflow.__dict__, "workflow should have a monitor"
        if callable(workflow.monitor):
            monitor = workflow.monitor()
        else:
            monitor = workflow.monitor
        
        # JIT workflow step
        state_step = use_state(lambda: workflow.step)
        vmap_state_step = vmap(state_step)
        init_state = vmap_state_step.init_state(self.num_instances)
        self._workflow_step_: torch.jit.ScriptFunction = jit(
            vmap_state_step, trace=True, example_inputs=(init_state,)
        )
        # if no init step
        if type(workflow).init_step == Workflow.init_step:
            self._init_state_ = init_state
            self._workflow_init_step_ = self._workflow_step_
            return
        # otherwise, JIT workflow init step
        state_init_step = use_state(lambda: workflow.init_step)
        vmap_state_init_step = vmap(state_init_step)
        self._init_state_ = vmap_state_init_step.init_state(self.num_instances)
        self._workflow_init_step_: torch.jit.ScriptFunction = jit(
            vmap_state_init_step, trace=True, example_inputs=(self._init_state_,)
        )

    def evaluate(self, hyper_parameters: Dict[str, nn.Parameter]):
        # hyper parameters check
        for k in hyper_parameters.keys():
            assert (
                k in self._init_state_
            ), f"`{k}` should be in state dict of workflow and is `torch.nn.Parameter`"
            assert isinstance(
                self._init_state_[k], nn.Parameter
            ), f"`{k}` should correspond to a `torch.nn.Parameter`, got {type(self._init_state_[k])}"
        state = {**self._init_state_, **hyper_parameters}
        state = self._workflow_init_step_(state)
        for _ in range(self.iterations - 1):
            state = self._workflow_step_(state)

from typing import Dict

import torch
from torch import nn
from core import Problem, Workflow, jit_class, use_state, vmap, jit


@jit_class
class HpoProblemWrapper(Problem):

    def __init__(self, iterations: int, num_instances: int):
        super().__init__()
        assert iterations > 0, f"`iterations` should be greater than 0, got {iterations}"
        assert num_instances > 0, f"`num_instances` should be greater than 0, got {num_instances}"
        self.iterations = iterations
        self.num_instances = num_instances

    def setup(
        self,
        workflow: Workflow
    ):
        self.workflow = workflow
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
        

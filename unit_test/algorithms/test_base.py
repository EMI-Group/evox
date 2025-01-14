from unittest import TestCase

import torch

from evox.core import Algorithm, Problem, jit, use_state, vmap
from evox.workflows import EvalMonitor, StdWorkflow


class Sphere(Problem):
    def __init__(self):
        super().__init__()

    def evaluate(self, pop: torch.Tensor):
        return (pop**2).sum(-1)


class TestBase(TestCase):
    def run_algorithm(self, algo: Algorithm):
        monitor = EvalMonitor(full_fit_history=False, full_sol_history=False)
        prob = Sphere()
        workflow = StdWorkflow()
        workflow.setup(algo, prob, monitor)
        workflow.init_step()
        self.assertIsNotNone(workflow.get_submodule("monitor").topk_fitness)
        for _ in range(3):
            workflow.step()

    def run_trace_algorithm(self, algo: Algorithm):
        monitor = EvalMonitor(full_fit_history=False, full_sol_history=False)
        prob = Sphere()
        workflow = StdWorkflow()
        workflow.setup(algo, prob, monitor)
        workflow.init_step()
        state_step = use_state(lambda: workflow.step)
        state = state_step.init_state()
        self.assertIsNotNone(state["self.algorithm._monitor_.topk_fitness"])
        jit_state_step = jit(state_step, trace=True, example_inputs=(state,))
        for _ in range(3):
            state = jit_state_step(state)

    def run_vmap_algorithm(self, algo: Algorithm):
        prob = Sphere()
        workflow = StdWorkflow()
        workflow.setup(algo, prob)
        state_step = use_state(lambda: workflow.step)
        vmap_state_step = vmap(state_step)
        state = vmap_state_step.init_state(3)
        vmap_state_step = jit(
            vmap_state_step,
            trace=True,
            lazy=False,
            example_inputs=(state,),
        )
        for _ in range(3):
            state = vmap_state_step(state)

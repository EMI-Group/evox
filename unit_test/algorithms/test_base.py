from unittest import TestCase

import torch

from evox.core import Algorithm, Problem, use_state, vmap
from evox.workflows import EvalMonitor, StdWorkflow


def transform_getitem_args(x: torch.Tensor, index_args):
    print(x, index_args)
    if isinstance(index_args, tuple):
        return (x, list(index_args))
    elif not isinstance(index_args, (list, tuple)):
        return (x, [index_args])
    return (x, index_args)


class Sphere(Problem):
    def __init__(self):
        super().__init__()

    def evaluate(self, pop: torch.Tensor):
        return (pop**2).sum(-1)


class TestBase(TestCase):
    def run_algorithm(self, algo: Algorithm):
        state_dict = algo.state_dict()
        monitor = EvalMonitor(full_fit_history=False, full_sol_history=False)
        prob = Sphere()
        workflow = StdWorkflow(algo, prob, monitor)
        workflow.init_step()
        self.assertIsNotNone(workflow.get_submodule("monitor").topk_fitness)
        for _ in range(3):
            workflow.step()
        # reset state
        algo.load_state_dict(state_dict)

    def run_compiled_algorithm(self, algo: Algorithm):
        state_dict = algo.state_dict()
        monitor = EvalMonitor(full_fit_history=False, full_sol_history=False)
        prob = Sphere()
        workflow = StdWorkflow(algo, prob, monitor)
        workflow.init_step()
        jit_state_step = torch.compile(workflow.step)
        for _ in range(3):
            jit_state_step()
        # reset state
        algo.load_state_dict(state_dict)

    def run_vmap_algorithm(self, algo: Algorithm):
        prob = Sphere()
        workflow = StdWorkflow(algo, prob)
        params, buffers = torch.func.stack_module_state([workflow] * 3)
        state = params | buffers

        vmap_state_init_step = vmap(use_state(workflow.init_step), randomness="different")
        vmap_state_init_step = torch.compile(vmap_state_init_step)
        vmap_state_step = vmap(use_state(workflow.step), randomness="different")
        vmap_state_step = torch.compile(vmap_state_step)
        state = vmap_state_init_step(state)
        for _ in range(3):
            state = vmap_state_step(state)

    def run_all(self, algo: Algorithm):
        self.run_algorithm(algo)
        self.run_compiled_algorithm(algo)
        self.run_vmap_algorithm(algo)

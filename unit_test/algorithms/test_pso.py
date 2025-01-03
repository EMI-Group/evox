from unittest import TestCase
import torch
from evox.core import vmap, Problem, use_state, jit
from evox.workflows import StdWorkflow
from evox.algorithms import PSO


class TestPSO(TestCase):
    def setUp(self):
        class Sphere(Problem):
            def __init__(self):
                super().__init__()

            def evaluate(self, pop: torch.Tensor):
                return (pop**2).sum(-1)

        self.algo = PSO(pop_size=10, lb=-10 * torch.ones(3), ub=10 * torch.ones(3))
        self.prob = Sphere()

    def test_pso(self):
        workflow = StdWorkflow()
        workflow.setup(self.algo, self.prob)
        workflow.init_step()
        workflow.step()
        state_step = use_state(lambda: workflow.step)
        vmap_state_step = vmap(state_step)
        print(vmap_state_step.init_state(2))
        state = state_step.init_state()
        jit_state_step = jit(state_step, trace=True, example_inputs=(state,))
        state = state_step.init_state()
        for _ in range(2):
            workflow.step()
        for _ in range(2):
            state = jit_state_step(state)

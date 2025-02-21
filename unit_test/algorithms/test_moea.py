from unittest import TestCase

import torch

from evox.algorithms import MOEAD, NSGA2, NSGA3, RVEA
from evox.core import Algorithm, jit, use_state, vmap
from evox.problems.numerical import DTLZ2
from evox.workflows import StdWorkflow


class MOTestBase(TestCase):
    def run_algorithm(self, algo: Algorithm):
        prob = DTLZ2(m=3)
        workflow = StdWorkflow()
        workflow.setup(algo, prob)
        workflow.init_step()
        for _ in range(3):
            workflow.step()

    def run_trace_algorithm(self, algo: Algorithm):
        prob = DTLZ2(m=3)
        workflow = StdWorkflow()
        workflow.setup(algo, prob)
        workflow.init_step()
        state_step = use_state(lambda: workflow.step)
        state = state_step.init_state()
        jit_state_step = jit(state_step, trace=True, example_inputs=(state,))
        for _ in range(3):
            state = jit_state_step(state)

    def run_vmap_algorithm(self, algo: Algorithm):
        prob = DTLZ2(m=3)
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


class TestMOVariants(MOTestBase):
    def setUp(self):
        pop_size = 100
        dim = 12
        lb = -torch.ones(dim)
        ub = torch.ones(dim)
        self.algo = [
            NSGA2(pop_size=pop_size, n_objs=3, lb=lb, ub=ub),
            NSGA3(pop_size=pop_size, n_objs=3, lb=lb, ub=ub),
            RVEA(pop_size=pop_size, n_objs=3, lb=lb, ub=ub),
            MOEAD(pop_size=pop_size, n_objs=3, lb=lb, ub=ub),
        ]

    def test_moea_variants(self):
        for algo in self.algo:
            if isinstance(algo, MOEAD):
                self.run_algorithm(algo)
            elif isinstance(algo, NSGA3):
                self.run_algorithm(algo)
            else:
                self.run_algorithm(algo)
                self.run_trace_algorithm(algo)
                # self.run_vmap_algorithm(algo)

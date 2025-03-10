from unittest import TestCase

import torch

from evox.algorithms import MOEAD, NSGA2, NSGA3, RVEA, HypE
from evox.core import Algorithm, compile, use_state, vmap
from evox.problems.numerical import DTLZ2
from evox.workflows import StdWorkflow


class MOTestBase(TestCase):
    def run_algorithm(self, algo: Algorithm):
        prob = DTLZ2(m=3)
        workflow = StdWorkflow(algo, prob)
        workflow.init_step()
        for _ in range(3):
            workflow.step()

    def run_compiled_algorithm(self, algo: Algorithm):
        prob = DTLZ2(m=3)
        workflow = StdWorkflow(algo, prob)
        workflow.init_step()
        jit_state_step = compile(workflow.step)
        for _ in range(3):
            jit_state_step()

    def run_vmap_algorithm(self, algo: Algorithm):
        prob = DTLZ2(m=3)
        workflow = StdWorkflow(algo, prob)
        state_step = use_state(workflow.step)
        vmap_state_step = vmap(state_step, randomness="different")
        params, buffers = torch.func.stack_module_state([workflow] * 3)
        state = params | buffers
        vmap_state_step = compile(vmap_state_step)
        for _ in range(3):
            state = vmap_state_step(state)


class TestMOVariants(MOTestBase):
    def setUp(self):
        pop_size = 20
        dim = 10
        lb = -torch.ones(dim)
        ub = torch.ones(dim)
        self.algo = [
            NSGA2(pop_size=pop_size, n_objs=3, lb=lb, ub=ub),
            NSGA3(pop_size=pop_size, n_objs=3, lb=lb, ub=ub),
            RVEA(pop_size=pop_size, n_objs=3, lb=lb, ub=ub),
            MOEAD(pop_size=pop_size, n_objs=3, lb=lb, ub=ub),
            HypE(pop_size=pop_size, n_objs=3, lb=lb, ub=ub),
        ]

    def test_moea_variants(self):
        for algo in self.algo:
            self.run_algorithm(algo)
            self.run_compiled_algorithm(algo)
            # self.run_vmap_algorithm(algo)

import time
import torch
from torch.profiler import profile, ProfilerActivity
#
# import os
# import sys
# current_directory = os.getcwd()
# if current_directory not in sys.path:
#     sys.path.append(current_directory)

from src.evox.core import use_state, jit
from src.evox.workflows import StdWorkflow
from src.evox.algorithms import RVEA
from src.evox.problems.numerical import DTLZ2
from src.evox.metrics import igd
from src.evox.core import Algorithm, Problem, jit, use_state, vmap
from unittest import TestCase

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
        pop_size=100
        dim = 12
        lb = -torch.ones(dim)
        ub = torch.ones(dim)
        self.algo = [
            RVEA(pop_size=pop_size, n_objs=3, lb=lb, ub=ub)
        ]

    def test_rvea_variants(self):
        for algo in self.algo:
            self.run_algorithm(algo)
            self.run_trace_algorithm(algo)
            self.run_vmap_algorithm(algo)


if __name__ == "__main__":
    test = TestMOVariants()
    test.setUp()
    test.test_rvea_variants()
from unittest import TestCase, skipIf

import torch

from evox.algorithms import MOEAD, NSGA2, NSGA3, RVEA, HypE, RVEAa
from evox.core import Algorithm, compile, use_state, vmap
from evox.problems.numerical import DTLZ2
from evox.workflows import EvalMonitor, StdWorkflow


class MOTestBase(TestCase):
    def run_algorithm(self, algo: Algorithm):
        prob = DTLZ2(m=3)
        monitor = EvalMonitor(multi_obj=True, full_sol_history=True)
        workflow = StdWorkflow(algo, prob, monitor=monitor)
        workflow.init_step()
        for _ in range(3):
            workflow.step()

        monitor.get_pf()
        monitor.get_pf_fitness()

    def run_compiled_algorithm(self, algo: Algorithm):
        prob = DTLZ2(m=3)
        monitor = EvalMonitor(multi_obj=True, full_sol_history=True)
        workflow = StdWorkflow(algo, prob, monitor=monitor)
        workflow.init_step()
        jit_step = compile(workflow.step, dynamic=False)
        for _ in range(3):
            jit_step()

        monitor.get_pf()
        monitor.get_pf_fitness()

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
        torch.compiler.reset()
        torch.manual_seed(42)
        torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
        self.pop_size = 20
        self.dim = 10
        self.lb = -torch.ones(self.dim)
        self.ub = torch.ones(self.dim)

    def test_nsga2(self):
        algo = NSGA2(pop_size=self.pop_size, n_objs=3, lb=self.lb, ub=self.ub)
        self.run_algorithm(algo)
        self.run_compiled_algorithm(algo)

    def test_nsga3(self):
        algo = NSGA3(pop_size=self.pop_size, n_objs=3, lb=self.lb, ub=self.ub)
        self.run_algorithm(algo)
        self.run_compiled_algorithm(algo)

    def test_rvea(self):
        algo = RVEA(pop_size=self.pop_size, n_objs=3, lb=self.lb, ub=self.ub)
        self.run_algorithm(algo)
        self.run_compiled_algorithm(algo)

    def test_moead(self):
        algo = MOEAD(pop_size=self.pop_size, n_objs=3, lb=self.lb, ub=self.ub)
        self.run_algorithm(algo)
        self.run_compiled_algorithm(algo)

    @skipIf(
        torch.__version__.startswith("2.7."),
        "Torch 2.7 bug when running on non-AVX512 CPU: https://github.com/pytorch/pytorch/issues/152172",
    )
    def test_hype(self):
        algo = HypE(pop_size=self.pop_size, n_objs=3, lb=self.lb, ub=self.ub)
        self.run_algorithm(algo)
        self.run_compiled_algorithm(algo)

    def test_rveaa(self):
        algo = RVEAa(pop_size=self.pop_size, n_objs=3, lb=self.lb, ub=self.ub)
        self.run_algorithm(algo)
        self.run_compiled_algorithm(algo)

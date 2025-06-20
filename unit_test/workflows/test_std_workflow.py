import os
import unittest

import torch
import torch.multiprocessing as mp
import torch.nn as nn

from evox.core import Algorithm, Mutable, Problem, use_state, vmap
from evox.workflows import EvalMonitor, StdWorkflow


class BasicProblem(Problem):
    def __init__(self):
        super().__init__()
        self._eval_fn = vmap(BasicProblem._single_eval)

    def _single_eval(x: torch.Tensor, p: float = 2.0):
        return (x**p).sum()

    def evaluate(self, pop: torch.Tensor):
        return self._eval_fn(pop)


class BasicAlgorithm(Algorithm):
    def __init__(self, pop_size: int, lb: torch.Tensor, ub: torch.Tensor):
        super().__init__()
        assert lb.ndim == 1 and ub.ndim == 1, (
            f"Lower and upper bounds shall have ndim of 1, got {lb.ndim} and {ub.ndim}"
        )
        assert lb.shape == ub.shape, f"Lower and upper bounds shall have same shape, got {lb.ndim} and {ub.ndim}"
        self.pop_size = pop_size
        self.lb = lb
        self.ub = ub
        self.dim = lb.shape[0]
        self.pop = Mutable(torch.empty(self.pop_size, lb.shape[0], dtype=lb.dtype, device=lb.device))
        self.fit = Mutable(torch.empty(self.pop_size, dtype=lb.dtype, device=lb.device))

    def step(self):
        pop = torch.rand(self.pop_size, self.dim, dtype=self.lb.dtype, device=self.lb.device)
        pop = pop * (self.ub - self.lb)[None, :] + self.lb[None, :]
        self.pop = pop
        self.fit = self.evaluate(pop)


class TestStdWorkflow(unittest.TestCase):
    def setUp(self):
        torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
        self.algo = BasicAlgorithm(10, -10 * torch.ones(2), 10 * torch.ones(2))
        self.prob = BasicProblem()
        self.monitor = EvalMonitor()
        self.workflow = StdWorkflow(self.algo, self.prob, monitor=self.monitor)

    def test_basic_workflow(self):
        compiled_step = torch.compile(self.workflow.step)
        for _ in range(3):
            compiled_step()

    def test_vmap_workflow(self):
        state_step = use_state(self.workflow.step)
        vmap_state_step = vmap(state_step, randomness="different")
        params, buffers = torch.func.stack_module_state([self.workflow] * 3)
        state = params | buffers
        self.assertIsNotNone(state)
        compiled_state_step = torch.compile(vmap_state_step)
        self.assertIsNotNone(compiled_state_step(state))
        # Test if the monitor's history is a normal tensor (not batched)
        self.assertIsNotNone(self.monitor.fitness_history[0] * 1)

    def test_classic_workflow(self):
        class solution_transform(nn.Module):
            def forward(self, x: torch.Tensor):
                return x / 5

        class fitness_transform(nn.Module):
            def forward(self, f: torch.Tensor):
                return -f

        monitor = EvalMonitor(full_sol_history=True)
        workflow = StdWorkflow(
            self.algo,
            self.prob,
            monitor=monitor,
            solution_transform=solution_transform(),
            fitness_transform=fitness_transform(),
        )
        workflow.init_step()
        monitor = workflow.get_submodule("monitor")
        self.assertIsNotNone(monitor.topk_fitness)
        workflow.step()
        self.assertIsNotNone(monitor.topk_fitness)
        workflow.step()
        self.assertIsNotNone(monitor.topk_fitness)
        # test the plot function
        self.assertIsNotNone(monitor.plot())


def run_distributed_step(rank, world_size):
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    # the population size is 9, which is not divisible by 2
    # in this case, the last process will have a smaller population size
    algo = BasicAlgorithm(10, -10 * torch.ones(2), 10 * torch.ones(2))
    prob = BasicProblem()
    workflow = StdWorkflow(algo, prob, enable_distributed=True)
    torch.distributed.init_process_group(rank=rank, world_size=world_size)
    workflow.init_step()
    compiled_step = torch.compile(workflow.step)
    for _ in range(3):
        compiled_step()


class TestDistributedStdWorkflow(unittest.TestCase):
    def test_distributed_workflow(self):
        world_size = 2
        # set MASTER_ADDR and MASTER_PORT
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29501"
        mp.spawn(run_distributed_step, args=(world_size,), nprocs=world_size, join=True)

import os
import sys

current_directory = os.getcwd()
if current_directory not in sys.path:
    sys.path.append(current_directory)

import torch
from torch import nn

from src.core import jit_class, Problem, Algorithm, trace_impl, batched_random, Parameter
from src.workflows import StdWorkflow
from src.problems.hpo_wrapper import HPOProblemWrapper, HPOFitnessMonitor
from src.algorithms import PSO


if __name__ == "__main__":

    @jit_class
    class BasicProblem(Problem):

        def __init__(self):
            super().__init__()

        def evaluate(self, x: torch.Tensor):
            return (x * x).sum(-1)

    @jit_class
    class BasicAlgorithm(Algorithm):

        def __init__(self, pop_size: int):
            super().__init__()
            self.pop_size = pop_size
            self.hp = Parameter([1.0, 2.0])

        def setup(self, lb: torch.Tensor, ub: torch.Tensor):
            assert (
                lb.ndim == 1 and ub.ndim == 1
            ), f"Lower and upper bounds shall have ndim of 1, got {lb.ndim} and {ub.ndim}"
            assert (
                lb.shape == ub.shape
            ), f"Lower and upper bounds shall have same shape, got {lb.ndim} and {ub.ndim}"
            self.lb = lb
            self.ub = ub
            self.dim = lb.shape[0]
            self.pop = nn.Buffer(
                torch.empty(self.pop_size, lb.shape[0], dtype=lb.dtype, device=lb.device)
            )
            self.fit = nn.Buffer(torch.empty(self.pop_size, dtype=lb.dtype, device=lb.device))
            return self

        def step(self):
            pop = torch.rand(self.pop_size, self.dim, dtype=self.lb.dtype, device=self.lb.device)
            pop = pop * (self.ub - self.lb)[None, :] + self.lb[None, :]
            pop = pop * self.hp[0]
            self.pop.copy_(pop)
            self.fit.copy_(self.evaluate(pop))

        @trace_impl(step)
        def trace_step(self):
            pop = batched_random(
                torch.rand, self.pop_size, self.dim, dtype=self.lb.dtype, device=self.lb.device
            )
            pop = pop * (self.ub - self.lb)[None, :] + self.lb[None, :]
            pop = pop * self.hp[0]
            self.pop = pop
            self.fit = self.evaluate(pop)

    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    algo = BasicAlgorithm(pop_size=10)
    algo.setup(-10 * torch.ones(2), 10 * torch.ones(2))
    prob = BasicProblem()
    monitor = HPOFitnessMonitor()
    workflow = StdWorkflow()
    workflow.setup(algo, prob, monitor=monitor)

    hpo_prob = HPOProblemWrapper(iterations=9, num_instances=7)
    hpo_prob.setup(workflow)
    params = HPOProblemWrapper.extract_parameters(hpo_prob.init_state)
    print(params)
    params["self.algorithm.hp"] = torch.nn.Parameter(torch.rand(7, 2, device="cuda"), requires_grad=False)
    print(hpo_prob.evaluate(params))

    class solution_transform(torch.nn.Module):
        def forward(self, x: torch.Tensor):
            return {"self.algorithm.hp": x}
        
    pso = PSO(pop_size=7)
    pso.setup(
        -10 * torch.ones(2),
        10 * torch.ones(2),
    )
    outer_workflow = StdWorkflow()
    outer_workflow.setup(pso, hpo_prob, solution_transform=solution_transform())
    outer_workflow.init_step()
    print(outer_workflow.algorithm.local_best_fitness)
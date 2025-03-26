"""Benchmark the performance of PSO algorithm in EvoX."""

import time

import torch
from torch.profiler import ProfilerActivity, profile

from evox.algorithms import PSO
from evox.core import Problem, compile
from evox.workflows import StdWorkflow


def run_pso():
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    class Sphere(Problem):
        def __init__(self):
            super().__init__()

        def evaluate(self, pop: torch.Tensor):
            return (pop**2).sum(-1)

    pop_size = 100
    dim = 500
    algo = PSO(pop_size=pop_size, lb=-10 * torch.ones(dim), ub=10 * torch.ones(dim))
    prob = Sphere()

    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.get_default_device())
    workflow = StdWorkflow(algo, prob)
    workflow.init_step()
    workflow.step()
    t = time.time()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for _ in range(1000):
            workflow.step()
    print(time.time() - t)
    print(prof.key_averages().table(), flush=True)
    print("\n")
    torch.cuda.synchronize()

    compiled_step = compile(workflow.step)
    compiled_step()
    t = time.time()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for _ in range(1000):
            compiled_step()
    torch.cuda.synchronize()
    print(time.time() - t)
    print(prof.key_averages().table(), flush=True)
    print("\n")

    compiled_step = compile(workflow.step, mode="max-autotune-no-cudagraphs")
    compiled_step()
    t = time.time()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for _ in range(1000):
            compiled_step()
    torch.cuda.synchronize()
    print(time.time() - t)
    print(prof.key_averages().table(), flush=True)
    print("\n")


if __name__ == "__main__":
    run_pso()

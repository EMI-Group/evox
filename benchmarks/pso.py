"""Benchmark the performance of PSO algorithm in EvoX."""

import time
import torch
from torch.profiler import profile, ProfilerActivity
from evox.core import vmap, Problem, use_state, jit
from evox.workflows import StdWorkflow
from evox.algorithms import PSO


def run_pso():
    class Sphere(Problem):
        def __init__(self):
            super().__init__()

        def evaluate(self, pop: torch.Tensor):
            return (pop**2).sum(-1)

    algo = PSO(pop_size=10, lb=-10 * torch.ones(3), ub=10 * torch.ones(3))
    prob = Sphere()

    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.get_default_device())
    workflow = StdWorkflow()
    workflow.setup(algo, prob)
    workflow.init_step()
    workflow.step()
    state_step = use_state(lambda: workflow.step)
    vmap_state_step = vmap(state_step)
    print(vmap_state_step.init_state(2))
    state = state_step.init_state()
    jit_state_step = jit(state_step, trace=True, example_inputs=(state,))
    state = state_step.init_state()
    t = time.time()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for _ in range(1000):
            workflow.step()
    print(prof.key_averages().table())
    torch.cuda.synchronize()
    t = time.time()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for _ in range(1000):
            state = jit_state_step(state)
    print(prof.key_averages().table())
    torch.cuda.synchronize()
    print(time.time() - t)


if __name__ == "__main__":
    run_pso()

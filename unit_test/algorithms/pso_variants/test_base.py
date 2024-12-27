import time
import torch
from torch.profiler import profile, ProfilerActivity

import os
import sys
current_directory = os.getcwd()
if current_directory not in sys.path:
    sys.path.append(current_directory)

from src.core import Algorithm, Problem, use_state, jit, vmap
from src.workflows import StdWorkflow

class Sphere(Problem):
    def __init__(self):
        super().__init__()

    def evaluate(self, pop: torch.Tensor):
        return (pop**2).sum(-1)


def test(algo: Algorithm, print_path: str | None = None, test_trace: bool = True):
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.get_default_device())
    prob = Sphere()
    workflow = StdWorkflow()
    workflow.setup(algo, prob)
    workflow.init_step()
    workflow.step()
    # test trace step
    if test_trace:
        state_step = use_state(lambda: workflow.step)
        vmap_state_step = vmap(state_step)
        jit(vmap_state_step, trace=True, lazy=False, example_inputs=(vmap_state_step.init_state(3),))
        state = state_step.init_state()
        jit_state_step = jit(state_step, trace=True, example_inputs=(state,))
        state = state_step.init_state()
    # print
    if print_path is not None:
        with open(os.path.join(print_path, "script.md"), "w") as ff:
            ff.write(workflow.step.inlined_graph.__str__())
        if test_trace:
            with open(os.path.join(print_path, "trace.md"), "w") as ff:
                ff.write(jit_state_step.inlined_graph.__str__())
    # profile
    t = time.time()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True
    ) as prof:
        for _ in range(1000):
            workflow.step()
    print(prof.key_averages().table())
    torch.cuda.synchronize()
    if test_trace:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True
        ) as prof:
            for _ in range(1000):
                state = jit_state_step(state)
        print(prof.key_averages().table())
        torch.cuda.synchronize()
    print(time.time() - t)
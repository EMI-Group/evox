import time
import torch
from torch.profiler import profile, ProfilerActivity

import os
import sys
current_directory = os.getcwd()
if current_directory not in sys.path:
    sys.path.append(current_directory)

from src.core import vmap, Problem, use_state, jit
from src.workflows import StdWorkflow 
from src.algorithms import CLPSO


if __name__ == "__main__":
    class Sphere(Problem):
        def __init__(self):
            super().__init__()

        def evaluate(self, pop: torch.Tensor):
            return (pop**2).sum(-1)

    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.get_default_device())
    algo = CLPSO(pop_size=100000)
    algo.setup(lb=-10 * torch.ones(1000), ub=10 * torch.ones(1000))
    prob = Sphere()
    workflow = StdWorkflow()
    workflow.setup(algo, prob)
    workflow.init_step()
    workflow.__sync__()
    workflow.step()
    workflow.__sync__()
    # with open("tests/a.md", "w") as ff:
    #     ff.write(workflow.step.inlined_graph.__str__())
    state_step = use_state(lambda: workflow.step)
    vmap_state_step = vmap(state_step)
    print(vmap_state_step.init_state(2))
    state = state_step.init_state()
    jit_state_step = jit(state_step, trace=True, example_inputs=(state,))
    state = state_step.init_state()
    # with open("tests/b.md", "w") as ff:
    #     ff.write(jit_state_step.inlined_graph.__str__())
    t = time.time()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True
    ) as prof:
        for _ in range(1000):
            workflow.step()
        # for _ in range(1000):
        #     state = jit_state_step(state)
    print(prof.key_averages().table())
    torch.cuda.synchronize()
    print(time.time() - t)
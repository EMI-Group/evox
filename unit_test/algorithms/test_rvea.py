import time
import torch
from torch.profiler import profile, ProfilerActivity

import os
import sys
current_directory = os.getcwd()
if current_directory not in sys.path:
    sys.path.append(current_directory)

from src.core import use_state, jit
from src.workflows import StdWorkflow
from src.algorithms import RVEA
from src.problems import DTLZ2
from src.metrics import igd


if __name__ == "__main__":
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.get_default_device())

    prob = DTLZ2(m=3)
    pf = prob.pf()
    algo = RVEA(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12), pf=pf)
    workflow = StdWorkflow()
    workflow.setup(algo, prob)
    workflow.init_step()
    state_step = use_state(lambda: workflow.step)
    state = state_step.init_state()
    jit_state_step = jit(state_step, trace=True, example_inputs=(state,))
    with open("../../tests/b.md", "w") as ff:
        ff.write(jit_state_step.inlined_graph.__str__())
    with open("../../tests/a.md", "w") as ff:
        ff.write(workflow.step.inlined_graph.__str__())

    t = time.time()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True
    ) as prof:
        for i in range(100):
            state = jit_state_step(state)
    print(prof.key_averages().table())
    torch.cuda.synchronize()
    print(time.time() - t)

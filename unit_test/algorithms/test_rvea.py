import os
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile

from evox.algorithms import RVEA
from evox.core import jit, use_state
from evox.metrics import igd
from evox.problems.numerical import DTLZ2
from evox.workflows import StdWorkflow

current_directory = os.getcwd()
if current_directory not in sys.path:
    sys.path.append(current_directory)


if __name__ == "__main__":
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.get_default_device())

    prob = DTLZ2(m=3)
    pf = prob.pf()
    algo = RVEA(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
    workflow = StdWorkflow()
    workflow.setup(algo, prob)
    workflow.init_step()
    state_step = use_state(lambda: workflow.step)
    state = state_step.init_state()
    jit_state_step = jit(state_step, trace=True, example_inputs=(state,))
    directory = "../../tests"
    os.makedirs(directory, exist_ok=True)
    file_path1 = os.path.join(directory, "b.md")
    file_path2 = os.path.join(directory, "a.md")
    with open(file_path1, "w") as ff:
        ff.write(jit_state_step.inlined_graph.__str__())
    with open(file_path2, "w") as ff:
        ff.write(workflow.step.inlined_graph.__str__())

    t = time.time()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
        for i in range(100):
            state = jit_state_step(state)
            fit = state["self.algorithm.fit"]
            fit = fit[~torch.isnan(fit).any(dim=1)]
            print(igd(fit, pf))
    print(prof.key_averages().table())
    torch.cuda.synchronize()
    print(time.time() - t)

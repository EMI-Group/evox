import os
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile

from evox.algorithms import RVEAa as alg
from evox.metrics import igd
from evox.problems.numerical import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7
from evox.workflows import StdWorkflow

current_directory = os.getcwd()
if current_directory not in sys.path:
    sys.path.append(current_directory)

if __name__ == "__main__":
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.get_default_device())
    torch.manual_seed(42)
    for i in range(1):
        print(i)
        prob = DTLZ4(m=3)
        pf = prob.pf()
        algo = alg(pop_size=100, n_objs=3, lb=torch.zeros(12), ub=torch.ones(12))
        workflow = StdWorkflow(algo, prob)
        workflow.init_step()
        # jit_state_step = torch.compile(workflow.step, backend="eager")
        jit_state_step = torch.compile(workflow.step)
        with open("aaaaa.txt", "w+") as ff:
            print(torch._dynamo.explain(jit_state_step)(), file=ff, flush=True)
        t = time.time()
        # with profile(
        #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True
        # ) as prof:
        # Example with TorchScript:
        # torch.manual_seed(42)
        for i in range(100):
            jit_state_step()
            print(igd(workflow.algorithm.fit, pf))
        print("finished")
        # print(prof.key_averages().table())
        # torch.cuda.synchronize()
        print(time.time() - t)

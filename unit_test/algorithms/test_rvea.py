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
from src.algorithms import PSO, RVEA
from src.problems import DTLZ2
from src.utils import minimum, maximum, pairwise_euclidean_dist

def igd(objs: torch.Tensor, pf: torch.Tensor, p: int = 1):
    min_dis = pairwise_euclidean_dist(pf, objs).min(dim=1).values
    return (min_dis.pow(p).sum() / pf.shape[0]).pow(1 / p)


if __name__ == "__main__":
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.get_default_device())

    prob = DTLZ2(m=3)
    pf = prob.pf()
    algo = RVEA(pop_size=10000, n_objs=3, seed=42, pf=pf)
    algo.setup(lb=-torch.zeros(1000), ub=torch.ones(1000))
    workflow = StdWorkflow()
    workflow.setup(algo, prob)
    # prob.setup()


    # workflow.step()
    # workflow.__sync__()
    #
    # with open("tests/a.md", "w") as ff:
    #     ff.write(workflow.step.inlined_graph.__str__())
    state_init_step = use_state(lambda: workflow.init_step)
    state_step = use_state(lambda: workflow.step)
    state = state_step.init_state()
    ## state = {k: (v if v.ndim < 1 or v.shape[0] != algo.pop_size else v[:3]) for k, v in state.items()}
    jit_state_init_step = jit(state_init_step, trace=True, example_inputs=(state,))
    state = state_step.init_state()
    jit_state_step = jit(state_step, trace=True, example_inputs=(state,))
    # state = state_step.init_state()
    # workflow._use_init = False
    with open("../../src/algorithms/tests/b.md", "w") as ff:
        ff.write(jit_state_step.inlined_graph.__str__())
    with open("../../src/algorithms/tests/a.md", "w") as ff:
        ff.write(workflow.step.inlined_graph.__str__())
    # state_step = use_state(lambda: workflow.step)
    # state = state_step.init_state()
    # jit_state_step = jit(state_step, trace=True, example_inputs=(state,))
    t = time.time()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True
    ) as prof:
        # workflow.init_step()
        # for i in range(100):
        #     workflow.step()
        state = jit_state_init_step(state)
        for i in range(1):
            state = jit_state_step(state)
            # fit = state["self.algorithm.fit"]
            # fit = fit[~torch.isnan(fit).any(dim=1)]
            # print(igd(fit, pf))
        #     # print(state)
    print(prof.key_averages().table())
    torch.cuda.synchronize()
    print(time.time() - t)

import time
import os
import torch
from torch.profiler import profile, ProfilerActivity

from src.core import Problem, use_state, jit
from src.algorithms import PSO
from src.workflows import StdWorkflow


if __name__ == "__main__":
    class Sphere(Problem):
        def __init__(self):
            super().__init__()

        def evaluate(self, pop: torch.Tensor):
            return (pop**2).sum(-1)

    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.get_default_device())
    algo = PSO(pop_size=100000)
    algo.setup(lb=-10 * torch.ones(1000), ub=10 * torch.ones(1000))
    prob = Sphere()
    workflow = StdWorkflow()
    workflow.setup(algo, prob)
    workflow.step()
    workflow.__sync__()

    log_root = "./tests"
    os.makedirs(log_root, exist_ok=True)

    log_file_a = os.path.join(log_root, "a.md")
    with open(log_file_a, "w") as ff:
        ff.write(workflow.step.inlined_graph.__str__())
    print(f"Please see the result log at `{log_file_a}`.")

    state_step = use_state(lambda: workflow.step)
    state = state_step.init_state()
    ## state = {k: (v if v.ndim < 1 or v.shape[0] != algo.pop_size else v[:3]) for k, v in state.items()}
    jit_state_step = jit(state_step, trace=True, example_inputs=(state,))
    state = state_step.init_state()

    log_file_b = os.path.join(log_root, "b.md")
    with open(log_file_b, "w") as ff:
        ff.write(jit_state_step.inlined_graph.__str__())
    print(f"Please see the result log at `{log_file_b}`.")

    t = time.time()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True
    ) as prof:
        # for _ in range(1000):
        #     workflow.step()
        for _ in range(1000):
            state = jit_state_step(state)
    print(prof.key_averages().table())
    torch.cuda.synchronize()
    print(time.time() - t)

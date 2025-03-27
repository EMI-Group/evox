import os

import torch
from torch.profiler import ProfilerActivity, profile

from evox.core import Algorithm, Problem, compile, use_state, vmap
from evox.workflows import EvalMonitor, StdWorkflow


class Sphere(Problem):
    def __init__(self):
        super().__init__()

    def evaluate(self, pop: torch.Tensor):
        return (pop**2).sum(-1)


def test(
    algo: Algorithm,
    print_path: str | None = None,
    profiling: bool = True,
    test_compile: bool = True,
):
    print("Current device: ", torch.get_default_device())

    monitor = EvalMonitor(full_fit_history=False, full_sol_history=False)
    prob = Sphere()
    workflow = StdWorkflow(algo, prob, monitor)
    workflow.init_step()
    # test compile step
    if test_compile:
        state_step = use_state(workflow.step)
        compile_step = compile(workflow.step)
        vmap_state_step = vmap(state_step, randomness="different")
        vmap_state_step = compile(vmap_state_step, fullgraph=True)
    else:
        compile_step = workflow.step
    # print
    if print_path is not None:
        with open(os.path.join(print_path, "compile.md"), "w") as ff:
            ff.write(torch._dynamo.explain(compile_step)().__str__())
        if test_compile:
            with open(os.path.join(print_path, "compile_vmap.md"), "w") as ff:
                state = torch.func.stack_module_state([workflow] * 3)
                state = state[0] | state[1]
                ff.write(
                    torch._dynamo.explain(vmap_state_step)(state).__str__()
                )
    # profile
    print("Initial best fitness:", workflow.monitor.topk_fitness)
    compile_step()
    if profiling:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True
        ) as prof:
            for _ in range(1000):
                compile_step()
        print(prof.key_averages().table(), flush=True)
    else:
        for _ in range(1000):
            compile_step()
    print("Final best fitness:", workflow.monitor.topk_fitness)
    torch.cuda.synchronize()
    if test_compile:
        state = torch.func.stack_module_state([workflow] * 3)
        state = state[0] | state[1]
        vmap_state_step(state)
        print("Initial best fitness:", state["monitor.topk_fitness"])
        if profiling:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True
            ) as prof:
                for _ in range(1000):
                    state = vmap_state_step(state)
            print(prof.key_averages().table(), flush=True)
        else:
            for _ in range(1000):
                state = vmap_state_step(state)
        print("Final best fitness:", state["monitor.topk_fitness"])
        torch.cuda.synchronize()


if __name__ == "__main__":
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

    from evox.algorithms import PSO

    test(PSO(pop_size=1000, lb=-10 * torch.ones(50), ub=10 * torch.ones(50)))

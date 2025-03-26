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
    test_trace: bool = True,
):
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    print("Current device: ", torch.get_default_device())

    monitor = EvalMonitor(full_fit_history=False, full_sol_history=False)
    prob = Sphere()
    workflow = StdWorkflow(algo, prob)
    # test trace step
    if test_trace:
        state_step = use_state(lambda: workflow.step)
        state = state_step.init_state()
        compile_state_step = compile(state_step, trace=True, example_inputs=(state,))
        vmap_state_step = vmap(state_step)
        vmap_state_step = compile(
            vmap_state_step,
            trace=True,
            lazy=False,
            example_inputs=(vmap_state_step.init_state(3),),
        )
    # print
    if print_path is not None:
        with open(os.path.join(print_path, "script.md"), "w") as ff:
            ff.write(workflow.step.inlined_graph.__str__())
        if test_trace:
            with open(os.path.join(print_path, "trace.md"), "w") as ff:
                ff.write(compile_state_step.inlined_graph.__str__())
            with open(os.path.join(print_path, "vmap.md"), "w") as ff:
                ff.write(vmap_state_step.inlined_graph.__str__())
    # profile
    workflow = StdWorkflow(algo, prob, monitor)
    workflow.init_step()
    print("Initial best fitness:", workflow.get_submodule("monitor").topk_fitness)
    if profiling:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
            for _ in range(1000):
                workflow.step()
        print(prof.key_averages().table())
    else:
        for _ in range(1000):
            workflow.step()
    print("Final best fitness:", workflow.get_submodule("monitor").topk_fitness)
    torch.cuda.synchronize()
    if test_trace:
        workflow = StdWorkflow(algo, prob, monitor)
        workflow.init_step()
        state_step = use_state(lambda: workflow.step)
        state = state_step.init_state()
        print("Initial best fitness:", state["self.algorithm._monitor_.topk_fitness"])
        compile_state_step = compile(state_step, trace=True, example_inputs=(state,))
        if profiling:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True
            ) as prof:
                for _ in range(1000):
                    state = compile_state_step(state)
            print(prof.key_averages().table())
        else:
            for _ in range(1000):
                state = compile_state_step(state)
        print("Final best fitness:", state["self.algorithm._monitor_.topk_fitness"])
        torch.cuda.synchronize()

import os
import time
import unittest

# import test_meta
# import test_meta_2
import test_meta_3

# import test_meta_4
# import test_meta_5
import torch
import torch._dynamo
from torch.profiler import ProfilerActivity, profile

from evox.algorithms import CSO, DE, PSO
from evox.core import Algorithm, Problem, compile
from evox.metrics import igd
from evox.operators.sampling import uniform_sampling
from evox.problems.hpo_wrapper import HPOFitnessMonitor, HPOProblemWrapper
from evox.problems.numerical import DTLZ3 as PRO
from evox.workflows import EvalMonitor, StdWorkflow


class solution_transform(torch.nn.Module):
    def __init__(self, n_objs: int):
        super().__init__()
        self.n_objs = n_objs

    def forward(self, x: torch.Tensor):
        y = x.view(x.size(0), -1, self.n_objs)
        return {"algorithm.ref_vec_init": y}


class metric(torch.nn.Module):
    def __init__(self, pf: torch.Tensor):
        super().__init__()
        self.pf = pf

    def forward(self, x: torch.Tensor):
        return igd(x, self.pf)


class InnerCore(unittest.TestCase):
    def setUp(
        self,
        inner_algo: Algorithm,
        inner_prob: Problem,
        pop_size: int,
        n_objs: int,
        dimensions: int,
        inner_iterations: int,
        num_instances: int,
        num_repeats: int = 1,
    ):
        self.inner_algo = inner_algo(pop_size=pop_size, n_objs=n_objs, lb=torch.zeros(dimensions), ub=torch.ones(dimensions))
        self.inner_prob = inner_prob(d=dimensions, m=n_objs)
        self.inner_monitor = HPOFitnessMonitor(num_repeats=num_repeats, multi_obj_metric=metric(self.inner_prob.pf()))
        self.inner_workflow = StdWorkflow(self.inner_algo, self.inner_prob, monitor=self.inner_monitor)
        self.inner_workflow.init_step()
        self.hpo_prob = HPOProblemWrapper(
            iterations=inner_iterations,
            num_instances=num_instances,
            num_repeats=num_repeats,
            workflow=self.inner_workflow,
            copy_init_state=True,
        )


class OuterCore(unittest.TestCase):
    def setUp(
        self,
        outer_algo: Algorithm,
        num_instances: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        n_objs: int,
        hpo_prob: HPOProblemWrapper,
    ):

        if outer_algo in [CSO, PSO, DE]:
            self.outer_algo = outer_algo(
                pop_size=num_instances,
                lb=torch.zeros(lb, device=torch.get_default_device()),
                ub=torch.ones(ub, device=torch.get_default_device()),
            )
        self.outer_monitor = EvalMonitor(full_sol_history=False, topk=1,full_fit_history=False)
        self.outer_workflow = StdWorkflow(
            self.outer_algo, hpo_prob, monitor=self.outer_monitor, solution_transform=solution_transform(n_objs)
        )
        # self.outer_workflow.init_step()


if __name__ == "__main__":
    start_time = time.time()
    torch.set_default_device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.get_default_device()
    torch._dynamo.reset()
    torch.manual_seed(42)
    print(torch.get_default_device())

    # Parameters of the inner algorithm
    pop_size = 100
    n_objs = 3
    dimensions = 12

    # Parameters of the hpo problem
    inner_iterations = 500
    num_instances = 100
    num_repeats = 1

    # Iterations of the outer algorithm
    outer_iterations = 100

    # inner_algo = [test_meta2.InnerRVEAa, test_meta3.InnerRVEAa, test_meta4.InnerRVEAa, test_meta5.InnerRVEAa]
    # outer_algo = [PSO, PSO, test_meta4.OuterPSO, test_meta5.OuterPSO]
    inner_algo = [test_meta_3.InnerRVEAa]
    outer_algo = [PSO]
    sampling, _ = uniform_sampling(pop_size, n_objs)
    v = sampling.to()
    bound = [v.numel()]

    for j in range(len(inner_algo)):
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
            # Initialize the inner core
            inner_core = InnerCore()
            inner_core.setUp(inner_algo[j], PRO, pop_size, n_objs, dimensions, inner_iterations, num_instances, num_repeats)

            # Initialize the outer core
            outer_core = OuterCore()
            outer_core.setUp(outer_algo[j], num_instances, bound[j], bound[j], n_objs, inner_core.hpo_prob)

            # params = inner_core.hpo_prob.get_init_params()
            # print("init params:\n", params)

            print("Outer algorithm: ", outer_algo[j].__name__)
            print("Inner algorithm: ", inner_algo[j].__name__)
            print("Inner problem:",inner_core.inner_prob.__class__.__name__)
            # jit_state_step = compile(outer_core.outer_workflow.step, fullgraph=True)
            jit_state_step = compile(outer_core.outer_workflow.step, fullgraph=True, disable=False)
            # with open("aaa.txt", "w+") as ff:
            #     print(torch._dynamo.explain(jit_state_step)(), file=ff, flush=True)
            outer_monitor = outer_core.outer_workflow.get_submodule("monitor")
            print(f"The cost time before compiling: {time.time() - start_time: .4f}(s).")
            for i in range(outer_iterations):
                jit_state_step()
                print(f"The {i}th iteration and time elapsed: {time.time() - start_time: .4f}(s).")
                if i % 10 == 0:
                    print("result:\n", outer_monitor.topk_fitness)

            print(f"The {outer_iterations}th iteration and time elapsed: {time.time() - start_time: .4f}(s).")
            print("params:\n", outer_monitor.topk_solutions, "\n")
            print("result:\n", outer_monitor.topk_fitness)
        # torch.cuda.synchronize()
        # table = prof.key_averages().table()
        # print(table)
        # with open('table.txt', 'w') as f:
        #     f.write(table)
        # graph = outer_core.outer_workflow.algorithm.step.inlined_graph
        # print(graph)
        # with open('graph.txt', 'w') as f:
        #     f.write(graph.__str__())
        # print(outer_monitor.best_fitness)

        # params = inner_core.hpo_prob.get_init_params()

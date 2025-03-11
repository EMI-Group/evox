import json
import os
import time
import unittest

import test_meta
import test_meta2
import test_meta3
import test_meta4
import test_meta5
import torch
from tqdm import tqdm

from evox.algorithms import CSO, DE, PSO
from evox.core import Algorithm, Problem
from evox.metrics import igd
from evox.operators.sampling import uniform_sampling
from evox.operators.selection import non_dominate_rank
from evox.problems.hpo_wrapper import HPOFitnessMonitor, HPOProblemWrapper
from evox.problems.numerical import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7
from evox.workflows import EvalMonitor, StdWorkflow


class solution_transform(torch.nn.Module):
    def __init__(self, n_objs: int):
        super().__init__()
        self.n_objs = n_objs

    def forward(self, x: torch.Tensor):
        y = x.view(x.size(0), -1, self.n_objs)
        return {"self.algorithm.ref_vec_init": y}


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
        pop_size: int,
        problem: Problem,
        n_objs: int,
        dimensions: int,
        inner_iterations: int,
        num_instances: int,
        num_repeats: int = 1,
    ):
        self.inner_algo = inner_algo(pop_size=pop_size, n_objs=n_objs, lb=torch.zeros(dimensions), ub=torch.ones(dimensions))
        self.inner_prob = problem
        self.inner_monitor = HPOFitnessMonitor(multi_obj_metric=metric(self.inner_prob.pf()))
        self.inner_workflow = StdWorkflow()
        self.inner_workflow.setup(self.inner_algo, self.inner_prob, monitor=self.inner_monitor)
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
        outer_iterations: int,
        hpo_prob: HPOProblemWrapper,
    ):

        if outer_algo in [test_meta3.OuterPSO, test_meta4.OuterPSO]:
            self.outer_algo = outer_algo(
                pop_size=num_instances,
                lb=torch.zeros(lb, device=torch.get_default_device()),
                ub=torch.ones(ub, device=torch.get_default_device()),
                n_objs=n_objs,
            )
        elif outer_algo in [test_meta5.OuterPSO]:
            self.outer_algo = outer_algo(
                pop_size=num_instances,
                lb=torch.zeros(lb, device=torch.get_default_device()),
                ub=torch.ones(ub, device=torch.get_default_device()),
                n_objs=n_objs,
                max_gen=outer_iterations,
            )
        elif outer_algo in [CSO, PSO, DE]:
            self.outer_algo = outer_algo(
                pop_size=num_instances,
                lb=torch.zeros(lb, device=torch.get_default_device()),
                ub=torch.ones(ub, device=torch.get_default_device()),
            )
        self.outer_monitor = EvalMonitor(full_sol_history=False, topk=1)
        self.outer_workflow = StdWorkflow()
        self.outer_workflow.setup(
            self.outer_algo, hpo_prob, monitor=self.outer_monitor, solution_transform=solution_transform(n_objs)
        )
        self.outer_workflow.init_step()


def run(algorithm_name, problem, num_iter=100, d=12):

        run_time = []
        obj = []
        pop = []
        start = time.perf_counter()

        # Parameters of the inner algorithm
        pop_size = 100
        n_objs = 3
        dimensions = d

        # Parameters of the hpo problem
        inner_iterations = 100
        num_instances = 100
        num_repeats = 1

        # Iterations of the outer algorithm
        outer_iterations = 100

        # inner_algo = [test_meta2.InnerRVEAa, test_meta3.InnerRVEAa, test_meta4.InnerRVEAa, test_meta5.InnerRVEAa]
        # outer_algo = [PSO, PSO, test_meta4.OuterPSO, test_meta5.OuterPSO]
        inner_algo = [test_meta3.InnerRVEAa]
        outer_algo = [PSO]
        sampling, _ = uniform_sampling(pop_size, n_objs)
        v = sampling.to()
        # bound = [v.numel() * 2, v.numel(), v.numel(), v.numel()]
        bound = [v.numel()]

        for j in range(len(inner_algo)):
            # Initialize the inner core
            inner_core = InnerCore()
            inner_core.setUp(inner_algo[j], pop_size, problem, n_objs, dimensions, inner_iterations, num_instances, num_repeats)

            # Initialize the outer core
            outer_core = OuterCore()
            outer_core.setUp(outer_algo[j], num_instances, bound[j], bound[j], n_objs, outer_iterations, inner_core.hpo_prob)

            outer_monitor = outer_core.outer_workflow.get_submodule("monitor")
            start = time.perf_counter()
            for i in range(outer_iterations):
                outer_core.outer_workflow.step()
                now = time.perf_counter()
                duration = now - start
                run_time.append(duration)
            obj.append(outer_monitor.topk_fitness.tolist())
            pop.append(outer_monitor.topk_solutions)

        return pop, obj, run_time


def evaluate(x, f, pf, alpha, num_iter=100):
    # m = pf.size(1)
    # ref = torch.ones(m,)
    IGD = []
    HV = []
    history_data = []
    for i in range(num_iter):
        # current_pop = x[i]
        current_obj = f[i]
        # print("current_obj",current_obj)
        # current_obj = current_obj[~torch.isnan(current_obj).all(dim=1)]
        # current_pop = current_pop[~torch.isnan(current_pop).all(dim=1)]
        # rank = non_dominate_rank(current_obj)
        # pf = rank == 0
        # pf_fitness = current_obj[pf]
        # pf_solutions = current_pop[pf]



        # IGD.append(igd(objs=pf_fitness,pf=pf))
        # fmax = torch.max(pf, dim=0)
        # f_hv = pf_fitness / (alpha * fmax)
        # f_hv = f_hv[torch.all(f_hv <= 1, dim=1)]
        # if f_hv.size(0) > 0:
        #     HV.append(hv(objs=pf_fitness,ref=ref))
        # else:
        #     HV.append(0)
        IGD.append(current_obj)
        # data = {"raw_pop": current_pop.tolist(), "raw_obj":current_obj.tolist(), "pf_solutions":pf_solutions.tolist(), "pf_fitness":pf_fitness.tolist()}
        data = {"pf_fitness":current_obj.tolist()}
        history_data.append(data)

    return history_data, IGD, HV


if __name__ == "__main__":

    torch.set_default_device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.get_default_device()

    # num_iter = 100

    algorithm_names = ["metaRVEAa"]
    problem_list = [
            DTLZ1(m=3),
            # DTLZ2(m=3),
            # DTLZ3(m=3),
            # DTLZ4(m=3),
            # DTLZ5(m=3),
            # DTLZ6(m=3),
            # DTLZ7(m=3),
    ]
    alpha_list = [1.5, 1.5, 50, 1.5, 5, 5, 5]
    num_runs = 1
    num_pro = 7

    experiment_stats = []

    directory = f"../data/model_performance"
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    for algorithm_name in algorithm_names:
        for j, problem in enumerate(problem_list):
            d = 7 if j == 0 else 12  # Dimension of the decision variables
            # d = 12  # Dimension of the decision variables
            print(f"Running {algorithm_name} on {problem.__class__.__name__} with dimension {d}")

            pf = problem.pf()
            for exp_id in tqdm(range(num_runs), desc=f"{algorithm_name} - Problem {problem.__class__.__name__}"):
                pop, obj, t = run(algorithm_name, problem, d=d)

                # history_data, igd_value, hv_value = evaluate(
                #     pop, obj, pf, alpha_list[j]
                # )

                # data = {"history_data": history_data, "igd": igd_value, "hv": hv_value, "time": t}
                data = {"igd": obj, "time": t}
                with open(f"{directory}/{algorithm_name}_{problem.__class__.__name__}_exp{exp_id}.json", "w") as f:
                    json.dump(data, f)

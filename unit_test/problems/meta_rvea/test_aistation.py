import json
import os
import time

import test_meta3
import torch
from tqdm import tqdm

from evox.algorithms import PSO, RVEAa
from evox.core import jit, use_state
from evox.metrics import igd
from evox.operators.sampling import uniform_sampling
from evox.problems.numerical import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7
from evox.workflows import StdWorkflow


def run(algorithm_name, problem, num_iter=100, d=12):

        run_time = []
        obj = []
        pop = []
        start = time.perf_counter()

        prob = problem
        pf = torch.tensor(prob.pf())  # 将 Pareto 前沿转换为张量并移动到 GPU
        algo = RVEAa(pop_size=100, n_objs=3, lb=-torch.zeros(d), ub=torch.ones(d))
        workflow = StdWorkflow()
        workflow.setup(algo, prob)
        workflow.init_step()
        state_step = use_state(lambda: workflow.step)
        state = state_step.init_state()
        jit_state_step = jit(state_step, trace=True, example_inputs=(state,))
        for j in range(100):
            state = jit_state_step(state)
            now = time.perf_counter()
            duration = now - start
            run_time.append(duration)
        fit = state["self.algorithm.fit"]
        pop1 = state["self.algorithm.pop"]
        obj.append(igd(fit, pf).tolist())
        pop.append(pop1)

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

    algorithm_names = ["RVEAa"]
    problem_list = [
            # DTLZ1(m=3),
            # DTLZ2(m=3),
            # DTLZ3(m=3),
            DTLZ4(m=3),
            # DTLZ5(m=3),
            # DTLZ6(m=3),
            # DTLZ7(m=3),
    ]
    alpha_list = [1.5, 1.5, 50, 1.5, 5, 5, 5]
    num_runs = 31
    num_pro = 7

    experiment_stats = []

    directory = f"../data/model_performance"
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    for algorithm_name in algorithm_names:
        for j, problem in enumerate(problem_list):
            # d = 7 if j == 0 else 12  # Dimension of the decision variables
            d = 12  # Dimension of the decision variables
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

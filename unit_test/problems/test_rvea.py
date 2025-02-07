import os
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile

from evox.algorithms import RVEA, RVEAa
from evox.core import jit, use_state
from evox.metrics import igd
from evox.problems.numerical import DTLZ7
from evox.workflows import StdWorkflow

current_directory = os.getcwd()
if current_directory not in sys.path:
    sys.path.append(current_directory)



if __name__ == "__main__":
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.get_default_device())

    output_file = "fit_history_with_pf.json"

    prob = DTLZ7(m=3)
    pf = torch.tensor(prob.pf())  # 将 Pareto 前沿转换为张量并移动到 GPU
    algo = RVEA(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
    workflow = StdWorkflow()
    workflow.setup(algo, prob)
    workflow.init_step()

    torch.manual_seed(42)
    state_step = use_state(lambda: workflow.step)
    state = state_step.init_state()
    jit_state_step = jit(state_step, trace=True, example_inputs=(state,))

    data = {
        "pareto_front": pf.tolist(),  # 将张量转换为列表以保存到 JSON 文件
        "generations": []
    }

    data["generations"].append({"generation": 0, "fitness": state["self.algorithm.fit"].tolist()})

    t = time.time()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True
    ) as prof:
        for i in range(100):
            state = jit_state_step(state)
            fit = state["self.algorithm.fit"]
            # fit = fit[~torch.any(torch.isnan(fit), dim=1)]

            data["generations"].append({"generation": i + 1, "fitness": fit.tolist()})
            print(f"Generation {i + 1} IGD: {igd(fit, pf)}")  # 计算 IGD

    # with open(output_file, "w") as file:
    #     json.dump(data, file, indent=4)

    print(prof.key_averages().table())
    # torch.cuda.synchronize()
    print(f"Total time: {time.time() - t} seconds")

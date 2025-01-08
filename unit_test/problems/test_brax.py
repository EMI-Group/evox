import time
import os
import sys

current_directory = os.getcwd()
if current_directory not in sys.path:
    sys.path.append(current_directory)

import torch
import torch.nn as nn

from src.utils import ParamsAndVector
from src.algorithms import PSO
from src.workflows import StdWorkflow, EvalMonitor
from src.problems.neuroevolution import BraxProblem


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(nn.Linear(11, 4), nn.Tanh(), nn.Linear(4, 3))

    def forward(self, x):
        x = self.features(x)
        return x


def test_model(model, device):
    inputs = torch.rand(1, 11, device=device)
    outputs = model(inputs)
    print("Test model output:", outputs)


def neuroevolution_process(
    workflow: StdWorkflow,
    adapter: ParamsAndVector,
    max_generation: int = 50,
) -> None:
    for index in range(max_generation):
        print(f"In generation {index}:")
        t = time.time()
        workflow.step()
        torch.cuda.synchronize()
        print(f"\tTime elapsed: {time.time() - t: .4f}(s).")
        monitor: EvalMonitor = workflow.get_submodule("monitor")
        print(f"\tTop fitness: {monitor.topk_fitness}")
        best_params = adapter.to_params(monitor.topk_solutions[0])
        print(f"\tBest params: {best_params}")


if __name__ == "__main__":
    # General setting
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Set random seed
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Initialize model
    model = SimpleCNN().to(device)
    for p in model.parameters():
        p.requires_grad = False
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of model parameters: {total_params}")
    test_model(model, device)
    print()

    # Initialize neuroevolution process
    adapter = ParamsAndVector(dummy_model=model)
    model_params = dict(model.named_parameters())
    pop_center = adapter.to_vector(model_params)
    lower_bound = pop_center - 1
    upper_bound = pop_center + 1

    # Population-based neuroevolution testing
    print("The population-based neuroevolution process start.")
    POP_SIZE = 1000
    problem = BraxProblem(
        policy=model,
        env_name="hopper",
        max_episode_length=1000,
        num_episodes=3,
        pop_size=POP_SIZE,
        device=device,
    )

    algorithm = PSO(
        pop_size=POP_SIZE,
        lb=lower_bound,
        ub=upper_bound,
        device=device,
    )
    algorithm.setup()

    pop_monitor = EvalMonitor(
        topk=3,
        device=device,  # choose the best three individuals
    )
    pop_monitor.setup()

    workflow = StdWorkflow(opt_direction="max")
    workflow.setup(
        algorithm=algorithm,
        problem=problem,
        solution_transform=adapter,
        monitor=pop_monitor,
        device=device,
    )
    neuroevolution_process(
        workflow=workflow,
        adapter=adapter,
        max_generation=3,
    )

    print("Tests completed.")

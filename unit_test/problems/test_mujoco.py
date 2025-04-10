import time
import unittest

import torch
import torch.nn as nn

from evox.algorithms import PSO
from evox.problems.neuroevolution.mujoco_playground import MujocoProblem
from evox.utils import ParamsAndVector
from evox.workflows import EvalMonitor, StdWorkflow


class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.features = nn.Sequential(nn.Linear(25, 8), nn.Tanh(), nn.Linear(8, 5))

    def forward(self, x):
        x = self.features(x)
        return torch.tanh(x)


def test_model(model, device):
    inputs = torch.rand(1, 25, device=device)
    outputs = model(inputs)
    print("Test model output:", outputs)


class TestMujocoProblem(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        seed = 1234
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def test_mujoco_problem(self):
        model = SimpleMLP().to(self.device)
        for p in model.parameters():
            p.requires_grad = False
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of model parameters: {total_params}")
        test_model(model, self.device)
        print()

        adapter = ParamsAndVector(dummy_model=model)
        model_params = dict(model.named_parameters())
        pop_center = adapter.to_vector(model_params)
        lower_bound = torch.full_like(pop_center, -5)
        upper_bound = torch.full_like(pop_center, 5)

        POP_SIZE = 10
        problem = MujocoProblem(
            policy=model,
            env_name="SwimmerSwimmer6",
            max_episode_length=100,
            num_episodes=3,
            pop_size=POP_SIZE,
            device=self.device,
        )

        algorithm = PSO(
            pop_size=POP_SIZE,
            lb=lower_bound,
            ub=upper_bound,
            device=self.device,
        )

        pop_monitor = EvalMonitor(
            topk=3,
            device=self.device,
        )

        workflow = StdWorkflow(
            algorithm=algorithm,
            problem=problem,
            monitor=pop_monitor,
            opt_direction="max",
            solution_transform=adapter,
            device=self.device,
        )

        # Running for a fixed number of generations
        max_generation = 3
        for index in range(max_generation):
            print(f"In generation {index}:")
            t = time.time()
            workflow.step()
        print(f"\tTime elapsed: {time.time() - t: .4f}(s).")
        monitor: EvalMonitor = workflow.get_submodule("monitor")
        print(f"\tTop fitness: {monitor.topk_fitness}")
        best_params = adapter.to_params(monitor.topk_solutions[0])
        print(f"\tBest params: {best_params}")

        problem.visualize(best_params)

    def test_compiled_mujoco_problem(self):
        model = SimpleMLP().to(self.device)
        for p in model.parameters():
            p.requires_grad = False
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of model parameters: {total_params}")
        test_model(model, self.device)
        print()

        adapter = ParamsAndVector(dummy_model=model)
        model_params = dict(model.named_parameters())
        pop_center = adapter.to_vector(model_params)
        lower_bound = torch.full_like(pop_center, -5)
        upper_bound = torch.full_like(pop_center, 5)

        POP_SIZE = 5
        problem = MujocoProblem(
            policy=model,
            env_name="SwimmerSwimmer6",
            max_episode_length=10,
            num_episodes=3,
            pop_size=POP_SIZE,
            device=self.device,
        )

        algorithm = PSO(
            pop_size=POP_SIZE,
            lb=lower_bound,
            ub=upper_bound,
            device=self.device,
        )

        pop_monitor = EvalMonitor(
            topk=3,
            device=self.device,
        )

        workflow = StdWorkflow(
            algorithm=algorithm,
            problem=problem,
            monitor=pop_monitor,
            opt_direction="max",
            solution_transform=adapter,
            device=self.device,
        )

        compiled_step = torch.compile(workflow.step)

        for index in range(3):
            print(f"In generation {index}:")
            t = time.time()
            compiled_step()
        print(f"\tTime elapsed: {time.time() - t: .4f}(s).")
        monitor: EvalMonitor = workflow.get_submodule("monitor")
        print(f"\tTop fitness: {monitor.topk_fitness}")
        best_params = adapter.to_params(monitor.topk_solutions[0])
        print(f"\tBest params: {best_params}")

        problem.visualize(best_params)

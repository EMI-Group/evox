import os
import sys

sys.path.append(os.getcwd() + "/src")

import torch
from torch import nn
from core import Algorithm, jit_class, trace_impl, batched_random


@jit_class
class PSO(Algorithm):
    def __init__(self, pop_size: int, w: float = 0.6, phi_p: float = 2.5, phi_g: float = 0.8):
        super().__init__(pop_size)
        self.w = w
        self.phi_p = phi_p
        self.phi_g = phi_g

    def setup(self, lb: torch.Tensor, ub: torch.Tensor):
        assert lb.shape == ub.shape
        assert lb.ndim == 1 and ub.ndim == 1
        self.dim = lb.shape[0]
        # setup
        lb = lb[torch.newaxis, :]
        ub = ub[torch.newaxis, :]
        length = ub - lb
        population = torch.rand(self.pop_size, self.dim)
        population = length * population + lb
        velocity = torch.rand(self.pop_size, self.dim)
        velocity = 2 * length * velocity - length
        # write to self
        self.lb = lb
        self.ub = ub
        self.length = length
        # mutable
        self.population = nn.Buffer(population)
        self.velocity = nn.Buffer(velocity)
        self.local_best_location = nn.Buffer(population)
        self.local_best_fitness = nn.Buffer(torch.empty(self.pop_size).fill_(torch.inf))
        self.global_best_location = nn.Buffer(population[0])
        self.global_best_fitness = nn.Buffer(torch.tensor(torch.inf))

    def ask(self):
        return self.population

    def _calc_new(self, rg: torch.Tensor, rp: torch.Tensor, fitness: torch.Tensor):
        compare = self.local_best_fitness > fitness
        local_best_location = torch.where(
            compare[:, torch.newaxis], self.population, self.local_best_location
        )
        local_best_fitness = torch.minimum(self.local_best_fitness, fitness)

        all_fitness = torch.cat([torch.atleast_1d(self.global_best_fitness), fitness])
        all_population = torch.cat([self.global_best_location[torch.newaxis, :], self.population])
        global_best_index = torch.argmin(all_fitness)
        global_best_location = all_population[global_best_index]
        global_best_fitness = all_fitness[global_best_index]

        velocity = (
            self.w * self.velocity
            + self.phi_p * rp * (local_best_location - self.population)
            + self.phi_g * rg * (global_best_location - self.population)
        )
        population = self.population + velocity
        population = torch.clip(population, self.lb, self.ub)
        velocity = torch.clip(velocity, self.lb, self.ub)

        return (
            population,
            velocity,
            local_best_location,
            local_best_fitness,
            global_best_location,
            global_best_fitness,
        )

    def tell(self, fitness: torch.Tensor):
        rg = torch.rand(self.pop_size, self.dim, dtype=fitness.dtype, device=fitness.device)
        rp = torch.rand(self.pop_size, self.dim, dtype=fitness.dtype, device=fitness.device)
        compare = self.local_best_fitness > fitness
        torch.where(
            compare[:, torch.newaxis],
            self.population,
            self.local_best_location,
            out=self.local_best_location,
        )
        torch.minimum(self.local_best_fitness, fitness, out=self.local_best_fitness)
        best_new_index = torch.argmin(fitness)
        best_new_fitness = fitness[best_new_index]
        if best_new_fitness < self.global_best_fitness:
            self.global_best_fitness = best_new_fitness
            self.global_best_location = self.population[best_new_index]
        self.velocity = (
            self.w * self.velocity
            + self.phi_p * rp * (self.local_best_location - self.population)
            + self.phi_g * rg * (self.global_best_location - self.population)
        )
        self.population.add_(self.velocity)
        torch.clip(self.population, self.lb, self.ub, out=self.population)
        torch.clip(self.velocity, self.lb, self.ub, out=self.velocity)

    @trace_impl(tell)
    def trace_tell(self, fitness: torch.Tensor):
        rg = batched_random(
            torch.rand, self.pop_size, self.dim, dtype=fitness.dtype, device=fitness.device
        )
        rp = batched_random(
            torch.rand, self.pop_size, self.dim, dtype=fitness.dtype, device=fitness.device
        )
        (
            self.population,
            self.velocity,
            self.local_best_location,
            self.local_best_fitness,
            self.global_best_location,
            self.global_best_fitness,
        ) = self._calc_new(rg, rp, fitness)

    def init_ask(self):
        return self.ask()

    def init_tell(self, fitness: torch.Tensor):
        return self.tell(fitness)


if __name__ == "__main__":
    from core import vmap, Problem, Workflow, use_state, jit
    import time
    from torch.profiler import profile, record_function, ProfilerActivity
    
    class Sphere(Problem):

        def __init__(self):
            super().__init__(num_objective=1)

        def evaluate(self, pop: torch.Tensor):
            return (pop**2).sum(-1)

    @jit_class
    class BasicWorkflow(Workflow):

        def __init__(self, max_iterations: int | None = None):
            super().__init__()
            self.max_iterations = 0 if max_iterations is None else max_iterations
            self._use_init = True

        def setup(
            self,
            algorithm: Algorithm,
            problem: Problem,
            device: str | torch.device | int | None = None,
        ):
            algorithm.to(device=device)
            problem.to(device=device)
            self.algorithm = algorithm
            self.problem = problem
            self.generation = nn.Buffer(torch.zeros((), dtype=torch.int32, device=device))

        def step(self):
            population = self.algorithm.init_ask() if self._use_init else self.algorithm.ask()
            fitness = self.problem.evaluate(population)
            self.algorithm.init_tell(fitness) if self._use_init else self.algorithm.tell(fitness)
            self.generation.add_(1)
            self._use_init = False

        @trace_impl(step)
        def trace_step(self):
            population = self.algorithm.init_ask() if self._use_init else self.algorithm.ask()
            fitness = self.problem.evaluate(population)
            self.algorithm.init_tell(fitness) if self._use_init else self.algorithm.tell(fitness)
            self.generation = self.generation + 1
            self._use_init = False

        def loop(self, max_iterations: int | None = None):
            max_iterations = self.max_iterations if max_iterations is None else max_iterations
            for _ in range(max_iterations):
                self.step()

    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    algo = PSO(pop_size=100000)
    algo.setup(lb=-10 * torch.ones(1000), ub=10 * torch.ones(1000))
    prob = Sphere()
    workflow = BasicWorkflow()
    workflow.setup(algo, prob)
    workflow.loop(10)
    print(workflow.generation)
    with open("tests/a.md", 'w') as ff:
        ff.write(workflow.step.inlined_graph.__str__())
    # state_step = use_state(lambda: workflow.step)
    # jit_state_step = jit(state_step, trace=True, example_inputs=(state_step.init_state(),))
    # state = state_step.init_state()
    t = time.time()
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
    workflow.loop(1000)
    # print(prof.key_averages().table())
    torch.cuda.synchronize()
    print(time.time() - t)

import sys

sys.path.append(__file__ + "/../..")

import torch
from torch import nn
from core import Algorithm, jit_class, trace_impl, batched_random


def clamp(a: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
    lb = torch.relu(lb - a)
    ub = torch.relu(a - ub)
    return a + lb - ub


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
        lb = lb[None, :]
        ub = ub[None, :]
        length = ub - lb
        population = torch.rand(self.pop_size, self.dim)
        population = length * population + lb
        velocity = torch.rand(self.pop_size, self.dim)
        velocity = 2 * length * velocity - length
        # write to self
        self.lb = lb
        self.ub = ub
        # mutable
        self.population = nn.Buffer(population)
        self.velocity = nn.Buffer(velocity)
        self.local_best_location = nn.Buffer(population)
        self.local_best_fitness = nn.Buffer(torch.empty(self.pop_size).fill_(torch.inf))
        self.global_best_location = nn.Buffer(population[0])
        self.global_best_fitness = nn.Buffer(torch.tensor(torch.inf))

    def ask(self):
        return self.population

    def tell(self, fitness: torch.Tensor):
        compare = self.local_best_fitness - fitness
        self.local_best_location = torch.where(
            compare[:, None] > 0, self.population, self.local_best_location
        )
        self.local_best_fitness = self.local_best_fitness - torch.relu(compare)
        best_new_index = torch.argmin(fitness)
        best_new_fitness = fitness[best_new_index]
        if best_new_fitness < self.global_best_fitness:
            self.global_best_fitness = best_new_fitness
            self.global_best_location = self.population[best_new_index]
        rg = torch.rand(self.pop_size, self.dim, dtype=fitness.dtype, device=fitness.device)
        rp = torch.rand(self.pop_size, self.dim, dtype=fitness.dtype, device=fitness.device)
        velocity = (
            self.w * self.velocity
            + self.phi_p * rp * (self.local_best_location - self.population)
            + self.phi_g * rg * (self.global_best_location - self.population)
        )
        population = self.population + velocity
        self.population = clamp(population, self.lb, self.ub)
        self.velocity = clamp(velocity, self.lb, self.ub)

    @trace_impl(tell)
    def trace_tell(self, fitness: torch.Tensor):
        compare = self.local_best_fitness - fitness
        self.local_best_location = torch.where(
            compare[:, None] > 0, self.population, self.local_best_location
        )
        self.local_best_fitness = self.local_best_fitness - torch.relu(compare)

        all_fitness = torch.cat([torch.atleast_1d(self.global_best_fitness), fitness])
        all_population = torch.cat([self.global_best_location[None, :], self.population])
        global_best_index = torch.argmin(all_fitness)
        self.global_best_location = all_population[global_best_index]
        self.global_best_fitness = all_fitness[global_best_index]

        rg = batched_random(
            torch.rand, fitness.shape[0], self.dim, dtype=fitness.dtype, device=fitness.device
        )
        rp = batched_random(
            torch.rand, fitness.shape[0], self.dim, dtype=fitness.dtype, device=fitness.device
        )
        self.velocity = (
            self.w * self.velocity
            + self.phi_p * rp * (self.local_best_location - self.population)
            + self.phi_g * rg * (self.global_best_location - self.population)
        )
        self.population = self.population + self.velocity
        self.population = clamp(self.population, self.lb, self.ub)
        self.velocity = clamp(self.velocity, self.lb, self.ub)

    def init_tell(self, fitness: torch.Tensor):
        return self.tell(fitness)

    @trace_impl(init_tell)
    def trace_init_tell(self, fitness: torch.Tensor):
        return self.trace_tell(fitness)


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
    print(torch.get_default_device())
    algo = PSO(pop_size=100000)
    algo.setup(lb=-10 * torch.ones(1000), ub=10 * torch.ones(1000))
    prob = Sphere()
    workflow = BasicWorkflow(max_iterations=1000)
    workflow.setup(algo, prob)
    workflow.loop(10)
    workflow.__sync__()
    print(workflow.generation)
    with open("tests/a.md", "w") as ff:
        ff.write(workflow.step.inlined_graph.__str__())
    state_step = use_state(lambda: workflow.step)
    state = state_step.init_state()
    ## state = {k: (v if v.ndim < 1 or v.shape[0] != algo.pop_size else v[:3]) for k, v in state.items()}
    jit_state_step = jit(state_step, trace=True, example_inputs=(state,))
    state = state_step.init_state()
    with open("tests/b.md", "w") as ff:
        ff.write(jit_state_step.inlined_graph.__str__())
    t = time.time()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True
    ) as prof:
        workflow.loop()
        # for _ in range(1000):
        #     state = jit_state_step(state)
    print(prof.key_averages().table())
    torch.cuda.synchronize()
    print(time.time() - t)

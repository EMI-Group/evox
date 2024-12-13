import sys

sys.path.append(__file__ + "/../..")

import torch
from torch import nn
from core import Algorithm, jit_class, trace_impl, batched_random
from operators import ReferenceVectorGuided, Polynomial, SimulatedBinary, UniformSampling

def clamp(a: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
    lb = torch.relu(lb - a)
    ub = torch.relu(a - ub)
    return a + lb - ub


@jit_class
class RVEA(Algorithm):
    def __init__(self, pop_size: int, n_objs, seed=None, alpha=2, fr=0.1, max_gen=100,
        selection_op=None,
        mutation_op=None,
        crossover_op=None,):
        super().__init__(pop_size)

        if seed is None:
            seed = torch.seed()
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

        self.n_objs = n_objs
        self.alpha = alpha
        self.fr = fr
        self.max_gen = max_gen

        self.selection = selection_op
        self.mutation = mutation_op
        self.crossover = crossover_op

    def setup(self, lb: torch.Tensor, ub: torch.Tensor):
        assert lb.shape == ub.shape
        assert lb.ndim == 1 and ub.ndim == 1
        self.dim = lb.shape[0]
        # setup

        # write to self
        self.lb = lb
        self.ub = ub
        # mutable

        if self.selection is None:
            self.selection = ReferenceVectorGuided()
        if self.mutation is None:
            self.mutation = Polynomial((lb, ub))
        if self.crossover is None:
            self.crossover = SimulatedBinary()
        self.sampling = UniformSampling(self.pop_size, self.n_objs)

        v = self.sampling()[0]
        v0 = v
        self.pop_size = v.shape[0]
        lb = lb[None, :]
        ub = ub[None, :]
        length = ub - lb
        population = torch.rand(self.pop_size, self.dim)
        population = length * population + lb

        self.population = nn.Buffer(population)
        self.fitness = nn.Buffer(torch.empty((self.pop_size, self.n_objs)).fill_(torch.inf))
        self.offspring = nn.Buffer(population)
        self.offspring_fitness = nn.Buffer(torch.empty((self.pop_size, self.n_objs)).fill_(torch.inf))
        self.reference_vector = nn.Buffer(v)
        self.init_v = nn.Buffer(v0)
        self.gen = 0

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

    @trace_impl(ask)
    def trace_ask(self):
        self.offspring = self.population + batched_random(
            torch.rand, self.pop_size, self.dim, dtype=self.population.dtype, device=self.population.device
        )
        return self.offspring

    @trace_impl(tell)
    def trace_tell(self, fitness: torch.Tensor):
        all_fitness = torch.cat([self.fitness, fitness])
        all_population = torch.cat([self.offspring, self.population])
        self.population = all_population[torch.argsort(all_fitness)][5: self.pop_size+3]
        self.fitness = all_fitness[torch.argsort(all_fitness)][5: self.pop_size+3]

    def init_tell(self, fitness: torch.Tensor):
        self.fitness = fitness

    def init_ask(self):
        return self.population

    @trace_impl(init_ask)
    def trace_init_ask(self, fitness: torch.Tensor):
        return self.population

    @trace_impl(init_tell)
    def trace_init_tell(self, fitness: torch.Tensor):
        self.fitness = fitness


if __name__ == "__main__":
    from core import vmap, Problem, Workflow, use_state, jit
    import time
    from torch.profiler import profile, record_function, ProfilerActivity
    from problems import DTLZ2
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
    algo = RVEA(pop_size=100000)
    algo.setup(lb=-10 * torch.ones(1000), ub=10 * torch.ones(1000))
    prob = DTLZ2(m=3)
    workflow = BasicWorkflow(max_iterations=1000)
    workflow.setup(algo, prob)
    # workflow.loop(10)
    workflow.__sync__()
    print(workflow.generation)
    with open("../tests/a.md", "w") as ff:
        ff.write(workflow.step.inlined_graph.__str__())
    state_step = use_state(lambda: workflow.step)
    state = state_step.init_state()
    ## state = {k: (v if v.ndim < 1 or v.shape[0] != algo.pop_size else v[:3]) for k, v in state.items()}
    jit_state_step = jit(state_step, trace=True, example_inputs=(state,))
    state = state_step.init_state()
    with open("../tests/b.md", "w") as ff:
        ff.write(jit_state_step.inlined_graph.__str__())
    t = time.time()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True
    ) as prof:
        # workflow.loop()
        for _ in range(100):
            state = jit_state_step(state)
    print(prof.key_averages().table())
    torch.cuda.synchronize()
    print(time.time() - t)

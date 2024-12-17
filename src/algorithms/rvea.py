import sys

sys.path.append(__file__ + "/../..")

import torch
from torch import nn
from core import Algorithm, jit_class, trace_impl, batched_random
from operators import simulated_binary, uniform_sampling, polynomial_mutation # ReferenceVectorGuided,
from typing import Optional, Callable


def clamp(a: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
    lb = torch.relu(lb - a)
    ub = torch.relu(a - ub)
    return a + lb - ub


@jit_class
class RVEA(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, seed: Optional[int] = None, alpha: float = 2, fr: float = 0.1, max_gen: int = 100,
                 selection_op: Optional[Callable] = None,
                 mutation_op: Optional[Callable] = None,
                 crossover_op: Optional[Callable] = None):
        super().__init__()
        self.pop_size = pop_size

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
            self.mutation = polynomial_mutation
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


        self.pop = nn.Buffer(population)
        print(self.pop)
        self.fit = nn.Buffer(torch.empty((self.pop_size, self.n_objs)).fill_(torch.inf))
        self.offspring = nn.Buffer(population)
        self.offspring_fitness = nn.Buffer(torch.empty((self.pop_size, self.n_objs)).fill_(torch.inf))
        self.reference_vector = nn.Buffer(v)
        self.init_v = nn.Buffer(v0)
        self.gen = 0

    def step(self):
        mating_pool = torch.randint(0, self.pop.shape[0], (self.pop_size,))
        pop = self.pop[mating_pool]
        pop += torch.rand(self.pop_size, self.dim, dtype=self.lb.dtype, device=self.lb.device)
        merge_pop = torch.cat([self.pop, pop], dim=0)
        print(merge_pop)
        print(merge_pop.device)
        fit = self.evaluate(pop)
        merge_fit = torch.cat([self.fit, fit], dim=0)
        print(merge_fit)
        self.pop = merge_pop[5:self.pop_size+2]
        self.fit = merge_fit[5:self.pop_size+2]

    @trace_impl(step)
    def trace_step(self):
        pop = batched_random(
            torch.rand, self.pop_size, self.dim, dtype=self.lb.dtype, device=self.lb.device
        )
        pop = pop * (self.ub - self.lb)[None, :] + self.lb[None, :]
        self.pop = pop
        self.fit = self.evaluate(pop)


if __name__ == "__main__":
    from core import vmap, Problem, Workflow, use_state, jit
    import time
    from torch.profiler import profile, record_function, ProfilerActivity
    from problems import DTLZ2
    from workflows import StdWorkflow

    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.get_default_device())
    algo = RVEA(pop_size=100000)
    algo.setup(lb=-10 * torch.ones(1000), ub=10 * torch.ones(1000))
    prob = DTLZ2(m=3)
    workflow = StdWorkflow()
    workflow.setup(algo, prob)
    workflow.step()
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
        for i in range(5):
            workflow.step()
    print(prof.key_averages().table())
    torch.cuda.synchronize()
    print(time.time() - t)

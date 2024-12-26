import torch
from torch import nn
from typing import Optional, Callable

from ..core import Algorithm, jit_class, trace_impl, batched_random
from ..operators import simulated_binary, uniform_sampling, polynomial_mutation, ref_vec_guided
from ..utils import maximum, clamp, nanmin, nanmax, pairwise_euclidean_dist

def igd(objs: torch.Tensor, pf: torch.Tensor, p: int = 1):
    min_dis = pairwise_euclidean_dist(pf, objs).min(dim=1).values
    return (min_dis.pow(p).sum() / pf.shape[0]).pow(1 / p)

@jit_class
class RVEA(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, seed: Optional[int] = None, pf: torch.Tensor = None, alpha: float = 2, fr: float = 0.1, max_gen: int = 100,
                 selection_op: Optional[Callable] = None,
                 mutation_op: Optional[Callable] = None,
                 crossover_op: Optional[Callable] = None):
        super().__init__()
        self.pop_size = pop_size

        # if seed is None:
        #     seed = torch.seed()
        # self.generator = torch.Generator()
        # self.generator.manual_seed(seed)

        self.n_objs = n_objs
        self.alpha = alpha
        self.fr = fr
        self.max_gen = max_gen
        self.pf = pf

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
            self.selection = ref_vec_guided
        if self.mutation is None:
            self.mutation = polynomial_mutation
        if self.crossover is None:
            self.crossover = SimulatedBinary()
        self.sampling = UniformSampling(self.pop_size, self.n_objs)

        v = self.sampling()[0]
        v0 = v
        self.pop_size = v.shape[0]
        length = ub - lb
        population = torch.rand(self.pop_size, self.dim)
        population = length * population + lb


        self.pop = nn.Buffer(population)
        print(self.pop)
        self.fit = nn.Buffer(torch.empty((self.pop_size, self.n_objs)).fill_(torch.inf))
        self.reference_vector = nn.Buffer(v)
        self.init_v = nn.Buffer(v0)
        self.gen = 0

    def init_step(self):
        """
        Perform the first optimization step of the workflow.

        Calls the `init_step` of the algorithm if overwritten; otherwise, its `step` method will be invoked.
        """
        self.fit = self.evaluate(self.pop)

    def rv_adaptation(self, pop_obj, v0):
        max_vals = nanmax(pop_obj, dim=0)[0]
        min_vals = nanmin(pop_obj, dim=0)[0]
        return v0 * (max_vals - min_vals)

    def step(self):
        self.gen += 1
        mating_pool = torch.randint(0, self.pop.shape[0], (self.pop_size,))
        pop = self.pop[mating_pool]
        crossovered = self.crossover(pop)
        offspring = self.mutation(crossovered, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)
        off_fit = self.evaluate(offspring)
        merge_pop = torch.cat([self.pop, offspring], dim=0)
        merge_fit = torch.cat([self.fit, off_fit], dim=0)

        # print(merge_fit)
        survivor, survivor_fit = self.selection(merge_pop, merge_fit, self.reference_vector, torch.tensor((self.gen / self.max_gen) ** self.alpha, device=merge_fit.device))

        nan_mask_survivor = torch.isnan(survivor).any(dim=1)
        self.pop = survivor[~nan_mask_survivor]
        self.fit = survivor_fit[~nan_mask_survivor]

        if self.gen % (1 / self.fr) == 0:
            self.reference_vector = self.rv_adaptation(survivor_fit, self.init_v)

        print(self.gen)
        print(igd(self.fit, self.pf))


        # print(merge_pop)
        # print(merge_fit)



    @trace_impl(step)
    def trace_step(self):
        self.gen += 1
        no_nan_pop = ~torch.isnan(self.pop).all(dim=1)
        max_idx = torch.sum(no_nan_pop, dtype=torch.int32)
        # pop = self.pop[no_nan_pop]
        mating_pool = torch.randint(0, max_idx, (self.pop_size,))
        torch.nonzero(no_nan_pop)
        pop = self.pop[torch.nonzero(no_nan_pop)[mating_pool].squeeze()]
        crossovered = self.crossover(pop)
        offspring = self.mutation(crossovered, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)
        off_fit = self.evaluate(offspring)
        merge_pop = torch.cat([self.pop, offspring], dim=0)
        merge_fit = torch.cat([self.fit, off_fit], dim=0)

        self.pop, self.fit = self.selection(merge_pop, merge_fit, self.reference_vector,
                                                torch.tensor((self.gen / self.max_gen) ** self.alpha, device=merge_fit.device))

        # nan_mask_survivor = torch.isnan(survivor).any(dim=1)
        # self.pop = survivor[~nan_mask_survivor]
        # self.fit = survivor_fit[~nan_mask_survivor]

        if self.gen % (1 / self.fr) == 0:
            self.reference_vector = self.rv_adaptation(self.fit, self.init_v)
        # print(self.gen)
        # print(igd(self.fit, self.pf))


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
    state_step = use_state(lambda: workflow.step)
    state = state_step.init_state()
    jit_state_step = jit(state_step, trace=True, example_inputs=(state,))
    t = time.time()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True
    ) as prof:
        workflow.init_step()
        for i in range(1000):
            workflow.step()
        for i in range(100):
            state = jit_state_step(state)
    print(prof.key_averages().table())
    torch.cuda.synchronize()
    print(time.time() - t)

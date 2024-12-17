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
class CSO(Algorithm):
    def __init__(self, pop_size: int, phi: float = 0.0, mean = None,stdev = None):
        super().__init__()
        self.pop_size = pop_size
        self.phi = phi
        self.mean = mean
        self.stdev = stdev

    def setup(self, lb: torch.Tensor, ub: torch.Tensor):
        assert lb.shape == ub.shape
        assert lb.ndim == 1 and ub.ndim == 1
        self.dim = lb.shape[0]
        # setup
        lb = lb[None, :]
        ub = ub[None, :]
        length = ub - lb

        # write to self
        self.lb = lb
        self.ub = ub

        if self.mean is not None and self.stdev is not None:
            population = self.stdev * torch.randn(self.pop_size, self.dim)
            population = torch.clamp(population, min=self.lb, max=self.ub)
        else:
            population = torch.rand(self.pop_size, self.dim)
            population = population * (self.ub - self.lb) + self.lb
        velocity = torch.rand(self.pop_size, self.dim)
        velocity = 2 * length * velocity - length
        # mutable
        self.population = nn.Buffer(population)
        self.velocity = nn.Buffer(velocity)


    def _get_params(self):
        shuffle_idx = torch.randperm(self.pop_size, dtype=torch.int32, device=self.population.device)
        lambda1 = torch.rand(self.pop_size // 2, self.dim, device=self.population.device)
        lambda2 = torch.rand(self.pop_size // 2, self.dim, device=self.population.device)
        lambda3 = torch.rand(self.pop_size // 2, self.dim, device=self.population.device)
        return shuffle_idx, lambda1, lambda2, lambda3

    @trace_impl(_get_params)
    def _trace_get_params(self):
        shuffle_idx = batched_random(torch.randperm, self.pop_size, dtype=torch.int32, device=self.population.device)
        lambda1 = batched_random(torch.rand, self.pop_size // 2, self.dim, device=self.population.device)
        lambda2 = batched_random(torch.rand, self.pop_size // 2, self.dim, device=self.population.device)
        lambda3 = batched_random(torch.rand, self.pop_size // 2, self.dim, device=self.population.device)
        return shuffle_idx, lambda1, lambda2, lambda3

    def step(self):
        shuffle_idx, lambda1, lambda2, lambda3 = self._get_params()
        # inv_shuffle_idx = torch.argsort(shuffle_idx)
        pop = self.population[shuffle_idx]
        vec = self.velocity[shuffle_idx]
        center = torch.mean(self.population, dim=0)[None, :]
        fit = self.evaluate(pop)
        left_pop = pop[:self.pop_size//2]
        right_pop = pop[self.pop_size//2:]
        left_vec = vec[:self.pop_size//2]
        right_vec = vec[self.pop_size//2:]
        left_fit = fit[:self.pop_size//2]
        right_fit = fit[self.pop_size//2:]
        mask = (left_fit < right_fit)[:, None]
        
        left_velocity = torch.where(mask, left_vec, 
            lambda1 * right_vec
            + lambda2 * (right_pop - left_pop)
            + self.phi * lambda3 * (center - left_pop)
        )
        right_velocity = torch.where(mask, right_vec, 
            lambda1 * left_vec
            + lambda2 * (left_pop - right_pop)
            + self.phi * lambda3 * (center - right_pop)
        )
        
        left_pop = left_pop + left_velocity
        right_pop = right_pop + right_velocity
        pop = clamp(torch.cat([left_pop, right_pop]), self.lb, self.ub)
        vec = clamp(torch.cat([left_velocity, right_velocity]), self.lb, self.ub)

        self.population = pop
        self.velocity = vec




if __name__ == "__main__":
    import time
    from torch.profiler import profile, ProfilerActivity
    
    from core import vmap, Problem, use_state, jit
    from workflows import StdWorkflow

    class Sphere(Problem):

        def __init__(self):
            super().__init__()

        def evaluate(self, pop: torch.Tensor):
            return (pop**2).sum(-1)

    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.get_default_device())
    algo = CSO(pop_size=100000)
    algo.setup(lb=-10 * torch.ones(1000), ub=10 * torch.ones(1000))
    prob = Sphere()
    workflow = StdWorkflow()
    workflow.setup(algo, prob)
    workflow.step()
    workflow.__sync__()
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
        # for _ in range(1000):
        #     workflow.step()
        for _ in range(1000):
            state = jit_state_step(state)
    print(prof.key_averages().table())
    torch.cuda.synchronize()
    print(time.time() - t)

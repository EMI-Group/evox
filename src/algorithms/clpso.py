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
class CLPSO(Algorithm):
    def __init__(self,
        pop_size,  # population size
        inertia_weight,  # w
        const_coefficient,  # c
        learning_probability,  # P_c. shape:(pop_size,). It can be different for each particle
    ):
        super().__init__()
        self.pop_size = pop_size
        self.w = inertia_weight
        self.c = const_coefficient
        self.P_c = learning_probability

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

    def _get_global(self, fitness: torch.Tensor):
        random_coefficient = torch.rand(self.pop_size, self.dim, device=self.population.device)

        # ----------------- Update local_best -----------------
        compare = self.local_best_fitness > fitness
        self.local_best_location = torch.where(compare[:, None], self.population, self.local_best_location)
        self.local_best_fitness = torch.minimum(self.local_best_fitness, fitness)

        # ----------------- Update global_best -----------------
        self.global_best_fitness, global_best_index = torch.min(torch.cat([self.global_best_fitness.unsqueeze(0), fitness]), dim=0)
        self.global_best_location = torch.cat([self.global_best_location.unsqueeze(0), self.population])[global_best_index]

        # ------------------ Choose pbest ----------------------

        rand1_index = torch.floor(
            torch.rand(self.pop_size, device=self.population.device) * self.pop_size
        ).long()
        rand2_index = torch.floor(
            torch.rand(self.pop_size, device=self.population.device) * self.pop_size
        ).long()
        learning_index = torch.where(
            self.local_best_fitness[rand1_index] < self.local_best_fitness[rand2_index],
            rand1_index,
            rand2_index,
        )
        rand_possibility = torch.rand(self.pop_size, device=self.population.device).unsqueeze(1)
        return random_coefficient, learning_index, rand_possibility

    @trace_impl(_get_global)
    def _trace_get_global(self, fitness: torch.Tensor):
        random_coefficient = batched_random(torch.rand,self.pop_size, self.dim, device=self.population.device)

        # ----------------- Update local_best -----------------
        compare = self.local_best_fitness > fitness
        self.local_best_location = torch.where(compare[:, None], self.population, self.local_best_location)
        self.local_best_fitness = torch.minimum(self.local_best_fitness, fitness)

        # ----------------- Update global_best -----------------
        self.global_best_fitness, global_best_index = torch.min(torch.cat([self.global_best_fitness.unsqueeze(0), fitness]), dim=0)
        self.global_best_location = torch.cat([self.global_best_location.unsqueeze(0), self.population])[global_best_index]

        # ------------------ Choose pbest ----------------------

        rand1_index = torch.floor(
            batched_random(torch.rand,self.pop_size, device=self.population.device) * self.pop_size
        ).long()
        rand2_index = torch.floor(
             batched_random(torch.rand,self.pop_size, device=self.population.device) * self.pop_size
        ).long()
        learning_index = torch.where(
            self.local_best_fitness[rand1_index] < self.local_best_fitness[rand2_index],
            rand1_index,
            rand2_index,
        )
        rand_possibility =  batched_random(torch.rand,self.pop_size, device=self.population.device).unsqueeze(1)
        return random_coefficient, learning_index, rand_possibility

    def step(self):
        fitness = self.evaluate(self.population)
        random_coefficient, learning_index, rand_possibility = self._get_global(fitness) 
        learning_local_best = self.local_best_location[learning_index, :]
        rand_possibility = rand_possibility.expand(-1, self.dim)

        P_c = self.P_c.unsqueeze(1).expand(-1, self.dim)
        local_best = torch.where(rand_possibility < P_c, learning_local_best, self.local_best_location)

        # ------------------------------------------------------

        velocity = self.w * self.velocity + self.c * random_coefficient * (local_best - self.population)
        self.velocity = clamp(velocity, self.lb, self.ub)
        population = self.population + velocity
        self.population = clamp(population, self.lb, self.ub)


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
    algo = CLPSO(pop_size=100000,inertia_weight=0.5,const_coefficient=2,learning_probability=torch.full((100000,), 0.5))
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
        for _ in range(1000):
            workflow.step()
        # for _ in range(1000):
        #     state = jit_state_step(state)
    print(prof.key_averages().table())
    torch.cuda.synchronize()
    print(time.time() - t)
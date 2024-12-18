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
class DMSPSOEL(Algorithm):
    def __init__(
            self, 
            dynamic_sub_swarm_size: int,  # one of the dynamic sub-swarms size
            dynamic_sub_swarms_num: int,  # number of dynamic sub-swarms
            following_sub_swarm_size: int,  # following sub-swarm size
            regrouped_iteration_num: int,  # number of iterations for regrouping
            max_iteration: int,  # maximum number of iterations
            inertia_weight: float,  # w
            pbest_coefficient: float,  # c_pbest
            lbest_coefficient: float,  # c_lbest
            rbest_coefficient: float,  # c_rbest
            gbest_coefficient: float,  # c_gbest
            ):
        super().__init__()
        self.pop_size = (
            dynamic_sub_swarm_size * dynamic_sub_swarms_num + following_sub_swarm_size
        )
        self.dynamic_sub_swarm_size = dynamic_sub_swarm_size
        self.dynamic_sub_swarms_num = dynamic_sub_swarms_num
        self.following_sub_swarm_size = following_sub_swarm_size
        self.regrouped_iteration_num = regrouped_iteration_num
        self.max_iteration = max_iteration
        self.w = inertia_weight
        self.c_pbest = pbest_coefficient
        self.c_lbest = lbest_coefficient
        self.c_rbest = rbest_coefficient
        self.c_gbest = gbest_coefficient

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
        self.iteration = 0
        dynamic_swarms = population[
            : self.dynamic_sub_swarm_size * self.dynamic_sub_swarms_num, :
        ]
        dynamic_swarms = dynamic_swarms.reshape(
            self.dynamic_sub_swarms_num, self.dynamic_sub_swarm_size, self.dim
        )
        local_best_location = dynamic_swarms[:, 0, :]
        local_best_fitness = torch.empty(self.dynamic_sub_swarms_num).fill_(torch.inf)
        # mutable
        self.population = nn.Buffer(population)
        self.velocity = nn.Buffer(velocity)
        self.personal_best_location = nn.Buffer(population)
        self.personal_best_fitness = nn.Buffer(torch.empty(self.pop_size).fill_(torch.inf))
        self.local_best_location = nn.Buffer(local_best_location)
        self.local_best_fitness = nn.Buffer(local_best_fitness)
        self.regional_best_index = nn.Buffer(torch.zeros(self.following_sub_swarm_size, dtype=torch.int))
        self.global_best_location = nn.Buffer(torch.zeros(self.dim))
        self.global_best_fitness = nn.Buffer(torch.tensor(torch.inf))

    def _get_rand1(self):
        rand_pbest = torch.rand(self.pop_size, self.dim, device=self.population.device)
        rand_lbest = torch.rand(self.dynamic_sub_swarms_num, self.dynamic_sub_swarm_size, self.dim, device=self.population.device)
        rand_rbest = torch.rand(self.following_sub_swarm_size, self.dim, device=self.population.device)
        return rand_pbest, rand_lbest, rand_rbest

    @trace_impl(_get_rand1)
    def _trace_get_rand1(self):
        rand_pbest = batched_random(torch.rand, self.pop_size, self.dim, device=self.population.device)
        rand_lbest = batched_random(torch.rand, self.dynamic_sub_swarms_num, self.dynamic_sub_swarm_size, self.dim, device=self.population.device)
        rand_rbest = batched_random(torch.rand, self.following_sub_swarm_size, self.dim, device=self.population.device)
        return rand_pbest, rand_lbest, rand_rbest
    
    def _get_rand2(self):
        rand_pbest = torch.rand(self.pop_size, self.dim, device=self.population.device)
        rand_gbest = torch.rand(self.pop_size, self.dim, device=self.population.device)
        return rand_pbest, rand_gbest

    @trace_impl(_get_rand2)
    def _trace_get_rand2(self):
        rand_pbest = batched_random(torch.rand, self.pop_size, self.dim, device=self.population.device)
        rand_gbest = batched_random(torch.rand, self.pop_size, self.dim, device=self.population.device)
        return rand_pbest, rand_gbest

    def step(self):
        fitness = self.evaluate(self.population)
        fitness = fitness.to(self.population.device)
        if self.iteration < 0.9 * self.max_iteration:
            self._update_strategy_1(fitness)
        else:
            self._update_strategy_2(fitness)
        self.iteration = self.iteration+1

    def _update_strategy_1(self, fitness):
        if self.iteration % self.regrouped_iteration_num == 0:
            self._regroup(fitness)
       
        # ----------------- Update personal_best -----------------
        better_fitness = self.personal_best_fitness > fitness
        personal_best_location = torch.where(
            better_fitness[:, None], self.population, self.personal_best_location
        )
        personal_best_fitness = torch.minimum(self.personal_best_fitness, fitness)

        # -----------------Update dynamic swarms -----------------
        dynamic_swarms_location = self.population[
            : self.dynamic_sub_swarm_size * self.dynamic_sub_swarms_num, :
        ].view(self.dynamic_sub_swarms_num, self.dynamic_sub_swarm_size, self.dim)
        
        dynamic_swarms_fitness = fitness[
            : self.dynamic_sub_swarm_size * self.dynamic_sub_swarms_num
        ].view(self.dynamic_sub_swarms_num, self.dynamic_sub_swarm_size)
        
        dynamic_swarms_velocity = self.velocity[
            : self.dynamic_sub_swarm_size * self.dynamic_sub_swarms_num, :
        ].view(self.dynamic_sub_swarms_num, self.dynamic_sub_swarm_size, self.dim)
        dynamic_swarms_pbest = personal_best_location[
            : self.dynamic_sub_swarm_size * self.dynamic_sub_swarms_num, :
        ].view(self.dynamic_sub_swarms_num, self.dynamic_sub_swarm_size, self.dim)

        # -----------------Update following swarm -----------------
        following_swarm_location = self.population[
            self.dynamic_sub_swarm_size * self.dynamic_sub_swarms_num :, :
        ]
        following_swarm_fitness = fitness[
            self.dynamic_sub_swarm_size * self.dynamic_sub_swarms_num :
        ]
        following_swarm_velocity = self.velocity[
            self.dynamic_sub_swarm_size * self.dynamic_sub_swarms_num :, :
        ]
        following_swarm_pbest = personal_best_location[
            self.dynamic_sub_swarm_size * self.dynamic_sub_swarms_num :, :
        ]

        # ----------------- Update local_best -----------------
        local_best_fitness, local_best_index = torch.min(dynamic_swarms_fitness, dim=1)  # shape:(dynamic_sub_swarms_num,)
        local_best_location = dynamic_swarms_location[
            torch.arange(dynamic_swarms_location.shape[0]), local_best_index
        ]

        # ----------------- Update regional_best -----------------
        regional_best_location = self.population[self.regional_best_index, :]

        # ---------------------------------------------------------
        # Caculate Dynamic Swarms Velocity
        rand_pbest, rand_lbest, rand_rbest = self._get_rand1()

        dynamic_swarms_rand_pbest = rand_pbest[
            : self.dynamic_sub_swarm_size * self.dynamic_sub_swarms_num, :
        ].view(self.dynamic_sub_swarms_num, self.dynamic_sub_swarm_size, self.dim)
        
        dynamic_swarms_velocity = (
            self.w * dynamic_swarms_velocity
            + self.c_pbest
            * dynamic_swarms_rand_pbest
            * (dynamic_swarms_pbest - dynamic_swarms_location)
            + self.c_lbest
            * rand_lbest
            * (
                torch.broadcast_to(
                    local_best_location[:, None, :],
                    (self.dynamic_sub_swarms_num, self.dynamic_sub_swarm_size, self.dim) 
                )
                - dynamic_swarms_location
            )   
        )

        # ---------------------------------------------------------
        # Caculate Following Swarm Velocity
        folowing_swarm_rand_pbest = rand_pbest[
            self.dynamic_sub_swarm_size * self.dynamic_sub_swarms_num :, :
        ]

        following_swarm_velocity = (
            self.w * following_swarm_velocity
            + self.c_pbest
            * folowing_swarm_rand_pbest
            * (following_swarm_pbest - following_swarm_location)
            + self.c_rbest * rand_rbest * (regional_best_location - following_swarm_location)
        )

        # ---------------------------------------------------------
        # Update Population
        dynamic_swarms_velocity = dynamic_swarms_velocity.view(
            self.dynamic_sub_swarm_size * self.dynamic_sub_swarms_num, self.dim
        )
        velocity = torch.cat(
            [dynamic_swarms_velocity, following_swarm_velocity], dim=0
        )
        population = self.population + velocity

        self.population = clamp(population, self.lb, self.ub)
        self.velocity = clamp(velocity, self.lb, self.ub)
        self.personal_best_location = personal_best_location
        self.personal_best_fitness = personal_best_fitness
        self.local_best_location = local_best_location
        self.local_best_fitness = local_best_fitness
    


    def _regroup(self, fitness):
        sort_index = torch.argsort(fitness, dim=0)
        dynamic_swarm_population_index = sort_index[
            : self.dynamic_sub_swarm_size * self.dynamic_sub_swarms_num
        ]
        dynamic_swarm_population_index = torch.randperm(
             dynamic_swarm_population_index.shape[0], device=self.population.device
        )
        regroup_index = torch.cat(
            (
                dynamic_swarm_population_index,
                sort_index[self.dynamic_sub_swarm_size * self.dynamic_sub_swarms_num :],
            )
        )

        population = self.population[regroup_index]
        velocity = self.velocity[regroup_index]
        personal_best_location = self.personal_best_location[regroup_index]
        personal_best_fitness = self.personal_best_fitness[regroup_index]

        dynamic_swarm_fitness = fitness[
            : self.dynamic_sub_swarm_size * self.dynamic_sub_swarms_num
        ]
        regional_best_index = torch.argsort(dynamic_swarm_fitness, dim=0)[
            : self.following_sub_swarm_size
        ]

        self.population = population
        self.velocity = velocity
        self.personal_best_location = personal_best_location
        self.personal_best_fitness = personal_best_fitness
        self.regional_best_index = regional_best_index

    def _update_strategy_2(self, fitness):

        # ---------------------------------------------------------
        # Update personal_best
        better_fitness = self.personal_best_fitness > fitness
        personal_best_location = torch.where(
            better_fitness[:, None], self.population, self.personal_best_location
        )
        personal_best_fitness = torch.minimum(self.personal_best_fitness, fitness)

        # ---------------------------------------------------------
        # Update global_best
        global_best_location = personal_best_location[torch.argmin(personal_best_fitness, dim=0)]
        global_best_fitness,_ = torch.min(personal_best_fitness, dim=0)

        # ---------------------------------------------------------
        rand_pbest, rand_gbest = self._get_rand2()
        velocity = (
            self.w * self.velocity
            + self.c_pbest * rand_pbest
            * (personal_best_location - self.population)
            + self.c_gbest * rand_gbest
            * (global_best_location - self.population)
        )
        population = self.population + velocity

        self.population = clamp(population, self.lb, self.ub)
        self.velocity = velocity
        self.personal_best_location = personal_best_location
        self.personal_best_fitness = personal_best_fitness
        self.global_best_location = global_best_location
        self.global_best_fitness = global_best_fitness

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
    algo = DMSPSOEL(5, 2, 10, 5, 100, 0.5, 1.5, 1.5, 1.5, 1.5)
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
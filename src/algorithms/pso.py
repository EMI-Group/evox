import os
import sys
sys.path.append(os.path.abspath(__file__ + "/../.."))

import torch
from torch import nn

from utils import clamp
from core import Parameter, Algorithm, jit_class, trace_impl, batched_random


@jit_class
class PSO(Algorithm):
    """The basic Particle Swarm Optimization (PSO) algorithm.

    ## Class Methods

    * `__init__`: Initializes the PSO algorithm with given parameters (population size, inertia weight, cognitive weight, and social weight).
    * `setup`: Initializes the PSO algorithm with given lower and upper bounds for particle positions, and sets up initial population, velocity, and buffers for tracking best local and global positions and fitness values.
    * `step`: Performs a single optimization step using Particle Swarm Optimization (PSO), updating local best positions and fitness values, and adjusting velocity and positions based on inertia, cognitive, and social components.

    Note that the `evaluate` method is not defined in this class, it is a proxy function of `Problem.evaluate` set by workflow; therefore, it cannot be used in class methods other than `step`.
    """

    def __init__(self, pop_size: int, w: float = 0.6, phi_p: float = 2.5, phi_g: float = 0.8):
        """
        Initialize the PSO algorithm with the given parameters.

        Args:
            pop_size (`int`): The size of the population.
            w (`float`, optional): The inertia weight. Defaults to 0.6.
            phi_p (`float`, optional): The cognitive weight. Defaults to 2.5.
            phi_g (`float`, optional): The social weight. Defaults to 0.8.
        """

        super().__init__()
        self.pop_size = pop_size
        # Here, Parameter is used to indicate that these values are hyper-parameters
        # so that they can be correctly traced and vector-mapped
        self.w = Parameter(w)
        self.phi_p = Parameter(phi_p)
        self.phi_g = Parameter(phi_g)

    def setup(self, lb: torch.Tensor, ub: torch.Tensor):
        """
        Initialize the PSO algorithm with the given lower and upper bounds.

        This function sets up the initial population and velocity for the
        particles within the specified bounds. It also initializes buffers
        for tracking the best local and global positions and fitness values.

        Args:
            lb (`torch.Tensor`): The lower bounds of the particle positions.
                            Must be a 1D tensor.
            ub (`torch.Tensor`): The upper bounds of the particle positions.
                            Must be a 1D tensor.

        Raises:
            `AssertionError`: If the shapes of lb and ub do not match or if
                            they are not 1D tensors.
        """
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

    def _set_global_and_random(self, fitness: torch.Tensor):
        best_new_index = torch.argmin(fitness)
        best_new_fitness = fitness[best_new_index]
        if best_new_fitness < self.global_best_fitness:
            self.global_best_fitness = best_new_fitness
            self.global_best_location = self.population[best_new_index]
        rg = torch.rand(self.pop_size, self.dim, dtype=fitness.dtype, device=fitness.device)
        rp = torch.rand(self.pop_size, self.dim, dtype=fitness.dtype, device=fitness.device)
        return rg, rp

    @trace_impl(_set_global_and_random)
    def _trace_set_global_and_random(self, fitness: torch.Tensor):
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
        return rg, rp

    def step(self):
        """
        Perform a normal optimization step using PSO.

        This function evaluates the fitness of the current population, updates the
        local best positions and fitness values, and adjusts the velocity and
        positions of particles based on inertia, cognitive, and social components.
        It ensures that the updated positions and velocities are clamped within the
        specified bounds.

        The local best positions and fitness values are updated if the current
        fitness is better than the recorded local best. The global best position
        and fitness are determined using helper functions.

        The velocity is updated based on the weighted sum of the previous velocity,
        the cognitive component (personal best), and the social component (global
        best). The population positions are then updated using the new velocities.
        """

        fitness = self.evaluate(self.population)
        compare = self.local_best_fitness - fitness
        self.local_best_location = torch.where(
            compare[:, None] > 0, self.population, self.local_best_location
        )
        self.local_best_fitness = self.local_best_fitness - torch.relu(compare)

        rg, rp = self._set_global_and_random(fitness)
        velocity = (
            self.w * self.velocity
            + self.phi_p * rp * (self.local_best_location - self.population)
            + self.phi_g * rg * (self.global_best_location - self.population)
        )
        population = self.population + velocity
        self.population = clamp(population, self.lb, self.ub)
        self.velocity = clamp(velocity, self.lb, self.ub)

    def init_step(self):
        """Perform the first step of the PSO optimization.
        See `step` for more details.
        """

        fitness = self.evaluate(self.population)
        self.local_best_fitness = fitness
        self.local_best_location = self.population

        rg, _ = self._set_global_and_random(fitness)
        velocity = self.w * self.velocity + self.phi_g * rg * (
            self.global_best_location - self.population
        )
        population = self.population + velocity
        self.population = clamp(population, self.lb, self.ub)
        self.velocity = clamp(velocity, self.lb, self.ub)


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
    algo = PSO(pop_size=10)
    algo.setup(lb=-10 * torch.ones(3), ub=10 * torch.ones(3))
    prob = Sphere()
    workflow = StdWorkflow()
    workflow.setup(algo, prob)
    workflow.init_step()
    workflow.__sync__()
    workflow.step()
    workflow.__sync__()
    # with open("tests/a.md", "w") as ff:
    #     ff.write(workflow.step.inlined_graph.__str__())
    state_step = use_state(lambda: workflow.step)
    vmap_state_step = vmap(state_step)
    print(vmap_state_step.init_state(2))
    state = state_step.init_state()
    jit_state_step = jit(state_step, trace=True, example_inputs=(state,))
    state = state_step.init_state()
    # with open("tests/b.md", "w") as ff:
    #     ff.write(jit_state_step.inlined_graph.__str__())
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

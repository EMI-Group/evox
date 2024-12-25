import torch
from torch import nn

from ..utils import clamp
from ..core import Parameter, Algorithm, jit_class, trace_impl, batched_random


@jit_class
class CLPSO(Algorithm):
    """The basic Particle Swarm Optimization (CLPSO) algorithm.

    ## Class Methods

    * `__init__`: Initializes the CLPSO algorithm with given parameters (population size, inertia weight, cognitive weight, and social weight).
    * `setup`: Initializes the CLPSO algorithm with given lower and upper bounds for particle positions, and sets up initial population, velocity, and buffers for tracking best local and global positions and fitness values.
    * `step`: Performs a single optimization step using Particle Swarm Optimization (CLPSO), updating local best positions and fitness values, and adjusting velocity and positions based on inertia, cognitive, and social components.

    Note that the `evaluate` method is not defined in this class, it is a proxy function of `Problem.evaluate` set by workflow; therefore, it cannot be used in class methods other than `step`.
    
    """
    def __init__(self,
        pop_size: int,  # population size
        inertia_weight: float = 0.5,  # w
        const_coefficient: float = 1.5,  # c
        learning_probability: float = 0.05,  # P_c. shape:(pop_size,). It can be different for each particle
    ):
        """
        Initialize the CLPSO algorithm with the given parameters.

        Args:
            pop_size (`int`): The size of the population.
            w (`float`, optional): The inertia weight. Defaults to 0.5.
            c (`float`, optional): The cognitive weight. Defaults to 1.5.
            P_c (`float`, optional): The social weight. Defaults to 0.05.

        """
         
        super().__init__()
        self.pop_size = pop_size
        self.w = Parameter(inertia_weight)
        self.c = Parameter(const_coefficient)
        self.P_c = Parameter(learning_probability)

    def setup(self, lb: torch.Tensor, ub: torch.Tensor):
        """
        Initialize the CLPSO algorithm with the given lower and upper bounds.

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
        self.personal_best_location = nn.Buffer(population)
        self.personal_best_fitness = nn.Buffer(torch.empty(self.pop_size).fill_(torch.inf))
        self.global_best_location = nn.Buffer(population[0])
        self.global_best_fitness = nn.Buffer(torch.tensor(torch.inf))

    def _get_params(self, fitness: torch.Tensor):
        random_coefficient = torch.rand(self.pop_size, self.dim, device=self.population.device)
        rand1_index = torch.floor(
            torch.rand(self.pop_size, device=self.population.device) * self.pop_size
        ).long()
        rand2_index = torch.floor(
            torch.rand(self.pop_size, device=self.population.device) * self.pop_size
        ).long() 
        rand_possibility = torch.rand(self.pop_size, device=self.population.device)
        # print("rand_possibility.shape-1",rand_possibility.shape)
        # rand_possibility = rand_possibility.unsqueeze(1)
        # print("rand_possibility.shape0",rand_possibility.shape)
        return random_coefficient, rand1_index, rand2_index, rand_possibility

    @trace_impl(_get_params)
    def _trace_get_params(self, fitness: torch.Tensor):
        random_coefficient = batched_random(torch.rand,self.pop_size, self.dim, device=self.population.device)
        rand1_index = torch.floor(
            batched_random(torch.rand,self.pop_size, device=self.population.device) * self.pop_size
        ).long()
        rand2_index = torch.floor(
             batched_random(torch.rand,self.pop_size, device=self.population.device) * self.pop_size
        ).long()
        rand_possibility = batched_random(torch.rand, self.pop_size, device=self.population.device)
        # print("rand_possibility.shape1",rand_possibility.shape)
        # rand_possibility = rand_possibility.unsqueeze(1)

        return random_coefficient, rand1_index, rand2_index, rand_possibility

    def step(self):
        fitness = self.evaluate(self.population)
        random_coefficient, rand1_index, rand2_index, rand_possibility = self._get_params(fitness)
        # rand_possibility = rand_possibility.expand(-1, self.dim)
        # print("rand_possibility.shape2",rand_possibility.shape)
        learning_index = torch.where(
            self.personal_best_fitness[rand1_index] < self.personal_best_fitness[rand2_index],
            rand1_index,
            rand2_index,
        )

        # ----------------- Update personal_best -----------------
        compare = self.personal_best_fitness > fitness
        self.personal_best_location = torch.where(compare[:, None], self.population, self.personal_best_location)
        self.personal_best_fitness = torch.minimum(self.personal_best_fitness, fitness)

        # ----------------- Update global_best -----------------
        self.global_best_fitness, global_best_index = torch.min(torch.cat([self.global_best_fitness.unsqueeze(0), fitness]), dim=0)
        self.global_best_location = torch.cat([self.global_best_location.unsqueeze(0), self.population])[global_best_index]

        # ------------------ Choose personal_best ----------------------
        learning_personal_best = self.personal_best_location[learning_index, :]
        personal_best = torch.where((rand_possibility < self.P_c[None])[:, None], learning_personal_best, self.personal_best_location)

        # ------------------------------------------------------
        velocity = self.w * self.velocity + self.c * random_coefficient * (personal_best - self.population)
        self.velocity = clamp(velocity, self.lb, self.ub)
        population = self.population + velocity
        self.population = clamp(population, self.lb, self.ub)

    # def init_step(self):
    #     """Perform the first step of the PSO optimization.
    #     See `step` for more details.
    #     """

    #     fitness = self.evaluate(self.population)
    #     self.local_best_fitness = fitness
    #     self.local_best_location = self.population

    #     rg, _ = self._set_global_and_random(fitness)
    #     velocity = self.w * self.velocity + self.phi_g * rg * (
    #         self.global_best_location - self.population
    #     )
    #     population = self.population + velocity
    #     self.population = clamp(population, self.lb, self.ub)
    #     self.velocity = clamp(velocity, self.lb, self.ub)
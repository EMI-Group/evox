import torch
from torch import nn

from ..utils import clamp
from ..core import Parameter, Algorithm, jit_class, trace_impl


@jit_class
class PSO(Algorithm):
    """The basic Particle Swarm Optimization (PSO) algorithm.

    ## Class Methods

    * `__init__`: Initializes the PSO algorithm with given parameters (population size, lower and upper bounds, inertia weight, cognitive weight, and social weight).
    * `step`: Performs a single optimization step using Particle Swarm Optimization (PSO), updating local best positions and fitness values, and adjusting velocity and positions based on inertia, cognitive, and social components.

    Note that the `evaluate` method is not defined in this class, it is a proxy function of `Problem.evaluate` set by workflow; therefore, it cannot be used in class methods other than `step`.
    """

    def __init__(
        self,
        pop_size: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        w: float = 0.6,
        phi_p: float = 2.5,
        phi_g: float = 0.8,
        device: torch.device | None = None,
    ):
        """
        Initialize the PSO algorithm with the given parameters.

        Args:
            pop_size (`int`): The size of the population.
            w (`float`, optional): The inertia weight. Defaults to 0.6.
            phi_p (`float`, optional): The cognitive weight. Defaults to 2.5.
            phi_g (`float`, optional): The social weight. Defaults to 0.8.
            lb (`torch.Tensor`): The lower bounds of the particle positions. Must be a 1D tensor.
            ub (`torch.Tensor`): The upper bounds of the particle positions. Must be a 1D tensor.
            device (`torch.device`, optional): The device to use for the tensors. Defaults to None.

        Raises:
            `AssertionError`: If the shapes of lb and ub do not match or if they are not 1D tensors or of different data types or devices.
        """

        super().__init__()
        if device is None:
            device = torch.get_default_device()
        self.pop_size = pop_size
        # Here, Parameter is used to indicate that these values are hyper-parameters
        # so that they can be correctly traced and vector-mapped
        self.w = Parameter(w, device=device)
        self.phi_p = Parameter(phi_p, device=device)
        self.phi_g = Parameter(phi_g, device=device)
        # check
        assert lb.shape == ub.shape and lb.ndim == 1 and ub.ndim == 1
        assert lb.dtype == ub.dtype and lb.device == ub.device
        self.dim = lb.shape[0]
        # setup
        lb = lb[None, :].to(device)
        ub = ub[None, :].to(device)
        length = ub - lb
        population = torch.rand(self.pop_size, self.dim, device=device)
        population = length * population + lb
        velocity = torch.rand(self.pop_size, self.dim, device=device)
        velocity = 2 * length * velocity - length
        # write to self
        self.lb = lb
        self.ub = ub
        # mutable
        self.population = nn.Buffer(population)
        self.velocity = nn.Buffer(velocity)
        self.local_best_location = nn.Buffer(population)
        self.local_best_fitness = nn.Buffer(torch.empty(self.pop_size, device=device).fill_(torch.inf))
        self.global_best_location = nn.Buffer(population[0])
        self.global_best_fitness = nn.Buffer(torch.tensor(torch.inf, device=device))

    def _set_global(self, fitness: torch.Tensor):
        best_new_index = torch.argmin(fitness)
        best_new_fitness = fitness[best_new_index]
        if best_new_fitness < self.global_best_fitness:
            self.global_best_fitness = best_new_fitness
            self.global_best_location = self.population[best_new_index]

    @trace_impl(_set_global)
    def _trace_set_global(self, fitness: torch.Tensor):
        all_fitness = torch.cat([self.global_best_fitness.unsqueeze(0), fitness])
        all_population = torch.cat([self.global_best_location[None, :], self.population])
        global_best_index = torch.argmin(all_fitness)
        self.global_best_location = all_population[global_best_index]
        self.global_best_fitness = all_fitness[global_best_index]

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

        # During normal workflow, we use `torch.jit.script` by default,
        # and the function `_set_global(fitness)` is invoked;
        # however, in the case of `torch.jit.trace` (i.e. `jit(..., trace=True)`),
        # the function `_trace_set_global(fitness)` is invoked instead.
        self._set_global(fitness)
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

    def init_step(self):
        """Perform the first step of the PSO optimization.
        
        See `step` for more details.
        """
        fitness = self.evaluate(self.population)
        self.local_best_fitness = fitness
        self.local_best_location = self.population

        self._set_global(fitness)
        rg = torch.rand(self.pop_size, self.dim, dtype=fitness.dtype, device=fitness.device)
        velocity = self.w * self.velocity + self.phi_g * rg * (
            self.global_best_location - self.population
        )
        population = self.population + velocity
        self.population = clamp(population, self.lb, self.ub)
        self.velocity = clamp(velocity, self.lb, self.ub)

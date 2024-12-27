import torch
import torch.nn as nn

from ..utils import clamp
from ..core import Parameter, Algorithm, jit_class, trace_impl, batched_random
from .utils import get_distance_matrix
from .topology_utils import get_square_neighbour, get_full_neighbour, build_adjacancy_list_from_matrix

from typing import Literal

@jit_class
class FIPS(Algorithm):
    """The basic FIPS algorithm.

    ## Class Methods

    * `__init__`: Initializes the FIPS algorithm with given parameters.
    * `setup`: Initializes the FIPS algorithm with given lower and upper bounds for particle positions, and sets up initial population, velocity, and buffers for tracking best local and global positions and fitness values.
    * `step`: Performs a single optimization step using FIPS, updating local best positions and fitness values, and adjusting velocity and positions based on inertia, cognitive, and social components.

    Note that the `evaluate` method is not defined in this class, it is a proxy function of `Problem.evaluate` set by workflow; therefore, it cannot be used in class methods other than `step`.
    """

    def __init__(
            self, 
            pop_size: int, 
            max_phi: float = 4.1,
            mean=None,
            stdev=None,
            topology: Literal[
                "Square", "Ring", "USquare", "URing", "All", "UAll"
            ] = "Square",
            weight_type: Literal["Constant", "Pbest", "Distance"] = "Distance",
            shortcut: int = 0,
        ):
        """
        Initialize the FIPS algorithm with the given parameters.

        Args:
            pop_size (`int`): The size of the population.
            max_phi (`float`, optional): The maximum phi value. Defaults to 4.1.
            mean (`float`, optional): The mean of the normal distribution. Defaults to None.
            stdev (`float`, optional): The standard deviation of the normal distribution. Defaults to None.
            topology (`str`, optional): The topology of the swarm. Defaults to "Square".
            weight_type (`str`, optional): The weight type of the swarm. Defaults to "Distance".
        """

        super().__init__()
        self.pop_size = pop_size
        # Here, Parameter is used to indicate that these values are hyper-parameters
        # so that they can be correctly traced and vector-mapped
        self.max_phi = Parameter(max_phi)
        self.mean = mean
        self.stdev = stdev
        self.topology = topology
        self.weight_type = weight_type
        self.shortcut = Parameter(shortcut)

    def setup(self, lb: torch.Tensor, ub: torch.Tensor):
        """
        Initialize the FIPS algorithm with the given lower and upper bounds.

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

        if self.mean is not None and self.stdev is not None:
            population = self.stdev * torch.randn(self.pop_size, self.dim)
            population = clamp(population, self.lb, self.ub)
            velocity = self.stdev * torch.randn(self.pop_size, self.dim)
        else:
            population = torch.rand(self.pop_size, self.dim)
            population = population * length + self.lb
            velocity = torch.rand(self.pop_size, self.dim)
            velocity = velocity * length * 2 - length

        if self.topology in ["Square", "USquare"]:
            adjacancy_matrix = get_square_neighbour(population)
        elif self.topology in ["All", "UAll"]:
            adjacancy_matrix = get_full_neighbour(population)
        else:
            raise NotImplementedError()

        phi = torch.ones(self.pop_size, 1) * self.max_phi
        # chi = 1-1/phi+\sqrt{\abs{phi**2-4*phi}}
        chi = 2 / (phi - 2 + torch.sqrt(torch.abs(phi * (phi - 4))))
        
        self.population = nn.Buffer(population)
        self.velocity = nn.Buffer(velocity)
        self.local_best_location = nn.Buffer(population)
        self.local_best_fitness = nn.Buffer(torch.empty(self.pop_size).fill_(torch.inf))
        self.neighbour_best_location = nn.Buffer(population)
        self.neighbour_best_fitness = nn.Buffer(torch.empty(self.pop_size).fill_(torch.inf))
        self.adjacancy_matrix = nn.Buffer(adjacancy_matrix)
        self.chi=nn.Buffer(chi)
        self.phi=nn.Buffer(phi)
        # self.global_best_location = nn.Buffer(population[0])
        # self.global_best_fitness = nn.Buffer(torch.tensor(torch.inf))

    def step(self):
        """
        Perform a normal optimization step using FIPS.

        This function performs a single optimization step using the FIPS algorithm.
        It updates the velocity and position of the particles, and updates the
        best local and global positions and fitness values.
        """

        fitness = self.evaluate(self.population)
        compare = self.local_best_fitness > fitness
        local_best_location = torch.where(
            compare[:, None], self.population, self.local_best_location
        )
        local_best_fitness = torch.minimum(self.local_best_fitness, fitness)

        adjacancy_matrix = self.adjacancy_matrix

        neighbour_list, neighbour_list_masking = build_adjacancy_list_from_matrix(
            adjacancy_matrix=adjacancy_matrix, keep_self_loop=True
        )

        if self.weight_type == "Constant":
            weight = self._calculate_weight_by_constant(adjacancy_list=neighbour_list)
        elif self.weight_type == "Pbest":
            weight = self._calculate_weight_by_fitness(
                fitness=local_best_fitness, adjacancy_list=neighbour_list
            )
        else:
            weight = self._calculate_weight_by_distance(
                location=local_best_location, adjacancy_list=neighbour_list
            )

        calculated_pm = self._get_PM(
            weight_list=weight,
            adjacancy_list=neighbour_list,
            adjacancy_list_mapping=neighbour_list_masking,
            location=local_best_location,
        )

        velocity = self.chi * (
            self.velocity + self.phi * (calculated_pm - self.population)
        )

        population = self.population + velocity
        population = clamp(population, self.lb, self.ub)

        self.population = population
        self.velocity = velocity
        self.local_best_location = local_best_location
        self.local_best_fitness = local_best_fitness

    def _get_PM(
        self, weight_list, adjacancy_list, adjacancy_list_mapping, location
    ):
        phik = self._set_random()
        phik = adjacancy_list_mapping[:,:,None] * phik * self.max_phi
        weight_phi = weight_list[:,:,None] * phik

        result = torch.stack(
            [self._calculate_pm(location, weight_phi[i], adjacancy_list[i]) for i in range(self.pop_size)]
        )
        return result
    
    def _set_random(self):
        phik = torch.rand(self.pop_size, self.pop_size, self.dim, device=self.population.device)
        return phik

    @trace_impl(_set_random)
    def _trace_set_random(self):
        phik = batched_random(torch.rand,self.pop_size, self.pop_size, self.dim, device=self.population.device)
        return phik
    
    def _calculate_pm(self, location: torch.Tensor, row_weight, row_adjacancy_list):
            upper = location[row_adjacancy_list] * row_weight
            lower = row_weight
            upper = upper.sum(dim=0)
            lower = lower.sum(dim=0)
            return upper / lower

    def _calculate_weight_by_constant(self, adjacancy_list):
        return torch.ones_like(adjacancy_list, dtype=torch.float32)

    def _calculate_weight_by_fitness(self, fitness, adjacancy_list):
        """
        each neighbor was weighted by the goodness of its previous best;
        goodness is set as 1/fitness.
        """
        weight = 1 / fitness[adjacancy_list]
        return weight

    def _calculate_weight_by_distance(self, location, adjacancy_list):
        distance_matrix = get_distance_matrix(location)
        distance_list = torch.stack(
            [
                distance_matrix[i, adjacancy_list[i]]
                for i in range(adjacancy_list.shape[0])
            ]
        )
        return distance_list
           
    
    # def init_step(self):
    #     """Perform the first step of the FIPS optimization.
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
    
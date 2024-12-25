import torch
from torch import nn

from ..utils import clamp
from ..core import Parameter, Algorithm, jit_class, trace_impl, batched_random


@jit_class
class DMSPSOEL(Algorithm):
    """The basic DMSPSOEL algorithm.

    ## Class Methods

    * `__init__`: Initializes the DMSPSOEL algorithm with given parameters.
    * `setup`: Initializes the DMSPSOEL algorithm with given lower and upper bounds for particle positions, and sets up initial population, velocity, and buffers for tracking best local and global positions and fitness values.
    * `step`: Performs a single optimization step using DMSPSOEL, updating local best positions and fitness values, and adjusting velocity and positions based on inertia, cognitive, and social components.
        

    Note that the `evaluate` method is not defined in this class, it is a proxy function of `Problem.evaluate` set by workflow; therefore, it cannot be used in class methods other than `step`.
    """
     
    def __init__(
            self, 
            dynamic_sub_swarm_size: int = 10,  # one of the dynamic sub-swarms size
            dynamic_sub_swarms_num: int = 5,  # number of dynamic sub-swarms
            following_sub_swarm_size: int = 10,  # following sub-swarm size
            regrouped_iteration_num: int = 50,  # number of iterations for regrouping
            max_iteration: int = 100,  # maximum number of iterations
            inertia_weight: float = 0.7,  # w
            pbest_coefficient: float = 1.5,  # c_pbest
            lbest_coefficient: float = 1.5,  # c_lbest
            rbest_coefficient: float = 1.0,  # c_rbest
            gbest_coefficient: float = 1.0,  # c_gbest
            ):
        """
        Initialize the DMSPSOEL algorithm with the given parameters.

        Args:
            dynamic_sub_swarm_size (`int`, optional): The size of the dynamic sub-swarm. Defaults to 10.
            dynamic_sub_swarms_num (`int`, optional): The number of dynamic sub-swarms. Defaults to 5.
            following_sub_swarm_size (`int`, optional): The size of the following sub-swarm. Defaults to 10.
            regrouped_iteration_num (`int`, optional): The number of iterations for regrouping. Defaults to 50.
            max_iteration (`int`, optional): The maximum number of iterations. Defaults to 100.
            inertia_weight (`float`, optional): The inertia weight. Defaults to 0.7.
            pbest_coefficient (`float`, optional): The cognitive weight. Defaults to 1.5.
            lbest_coefficient (`float`, optional): The social weight. Defaults to 1.5.
            rbest_coefficient (`float`, optional): The social weight. Defaults to 1.0.
            gbest_coefficient (`float`, optional): The social weight. Defaults to 1.0.
        """

        super().__init__()
        self.pop_size = (
            dynamic_sub_swarm_size * dynamic_sub_swarms_num + following_sub_swarm_size
        )
        self.dynamic_sub_swarm_size = Parameter(dynamic_sub_swarm_size)
        self.dynamic_sub_swarms_num = Parameter(dynamic_sub_swarms_num)
        self.following_sub_swarm_size = Parameter(following_sub_swarm_size)
        self.regrouped_iteration_num = Parameter(regrouped_iteration_num)
        self.max_iteration = Parameter(max_iteration)
        self.w = Parameter(inertia_weight)
        self.c_pbest = Parameter(pbest_coefficient)
        self.c_lbest = Parameter(lbest_coefficient)
        self.c_rbest = Parameter(rbest_coefficient)
        self.c_gbest = Parameter(gbest_coefficient)

    def setup(self, lb: torch.Tensor, ub: torch.Tensor):
        """
        
        Initialize the DMSPSOEL algorithm with the given lower and upper bounds.

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
        """
        Perform a single step of the DMSPSOEL algorithm.

        This function updates the population, velocity, personal best location,
        and personal best fitness based on the current fitness values. It also
        updates the local and global best positions and fitness values based on
        the dynamic sub-swarm and following sub-swarm. Finally, it updates the
        iteration count. 
        """
        
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
                local_best_location[:, None, :]
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

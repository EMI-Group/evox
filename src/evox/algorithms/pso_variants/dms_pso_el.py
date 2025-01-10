import torch

from ...core import Algorithm, Mutable, Parameter, jit_class, trace_impl
from ...utils import TracingCond, clamp


@jit_class
class DMSPSOEL(Algorithm):
    """The DMSPSOEL algorithm.

    ## Class Methods

    * `__init__`: Initializes the DMSPSOEL algorithm with given parameters.
    * `setup`: Initializes the DMSPSOEL algorithm with given lower and upper bounds for particle positions, and sets up initial population, velocity, and buffers for tracking best local and global positions and fitness values.
    * `step`: Performs a single optimization step using DMSPSOEL, updating local best positions and fitness values, and adjusting velocity and positions based on inertia, cognitive, and social components.


    Note that the `evaluate` method is not defined in this class, it is a proxy function of `Problem.evaluate` set by workflow; therefore, it cannot be used in class methods other than `step`.
    """

    # cSpell:words pbest lbest rbest gbest

    def __init__(
        self,
        lb: torch.Tensor,
        ub: torch.Tensor,
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
        device: torch.device | None = None,
    ):
        """
        Initialize the DMSPSOEL algorithm with the given parameters.

        :param lb: The lower bounds of the particle positions. Must be a 1D tensor.
        :param ub: The upper bounds of the particle positions. Must be a 1D tensor.
        :param dynamic_sub_swarm_size: The size of the dynamic sub-swarm. Defaults to 10.
        :param dynamic_sub_swarms_num: The number of dynamic sub-swarms. Defaults to 5.
        :param following_sub_swarm_size: The size of the following sub-swarm. Defaults to 10.
        :param regrouped_iteration_num: The number of iterations for regrouping. Defaults to 50.
        :param max_iteration: The maximum number of iterations. Defaults to 100.
        :param inertia_weight: The inertia weight. Defaults to 0.7.
        :param pbest_coefficient: The cognitive weight. Defaults to 1.5.
        :param lbest_coefficient: The social weight. Defaults to 1.5.
        :param rbest_coefficient: The social weight. Defaults to 1.0.
        :param gbest_coefficient: The social weight. Defaults to 1.0.
        :param device: The device to use for the tensors. Defaults to None.
        """
        super().__init__()
        device = torch.get_default_device() if device is None else device
        assert lb.shape == ub.shape and lb.ndim == 1 and ub.ndim == 1 and lb.dtype == ub.dtype
        self.dim = lb.shape[0]
        self.pop_size = dynamic_sub_swarm_size * dynamic_sub_swarms_num + following_sub_swarm_size
        self.dynamic_sub_swarm_size = dynamic_sub_swarm_size
        self.dynamic_sub_swarms_num = dynamic_sub_swarms_num
        self.following_sub_swarm_size = following_sub_swarm_size
        self.regrouped_iteration_num = Parameter(regrouped_iteration_num, device=device)
        self.max_iteration = Parameter(max_iteration, device=device)
        self.w = Parameter(inertia_weight, device=device)
        self.c_pbest = Parameter(pbest_coefficient, device=device)
        self.c_lbest = Parameter(lbest_coefficient, device=device)
        self.c_rbest = Parameter(rbest_coefficient, device=device)
        self.c_gbest = Parameter(gbest_coefficient, device=device)
        # setup
        lb = lb[None, :].to(device=device)
        ub = ub[None, :].to(device=device)
        length = ub - lb
        population = torch.rand(self.pop_size, self.dim, device=device)
        population = length * population + lb
        velocity = torch.rand(self.pop_size, self.dim, device=device)
        velocity = 2 * length * velocity - length
        # write to self
        self.lb = lb
        self.ub = ub
        self.iteration = Mutable(torch.tensor(0, dtype=torch.int32, device=device))
        dynamic_swarms = population[: self.dynamic_sub_swarm_size * self.dynamic_sub_swarms_num, :]
        dynamic_swarms = dynamic_swarms.reshape(self.dynamic_sub_swarms_num, self.dynamic_sub_swarm_size, self.dim)
        local_best_location = dynamic_swarms[:, 0, :]
        local_best_fitness = torch.empty(self.dynamic_sub_swarms_num, device=device).fill_(torch.inf)
        # mutable
        self.population = Mutable(population)
        self.velocity = Mutable(velocity)
        self.personal_best_location = Mutable(population)
        self.personal_best_fitness = Mutable(torch.empty(self.pop_size, device=device).fill_(torch.inf))
        self.local_best_location = Mutable(local_best_location)
        self.local_best_fitness = Mutable(local_best_fitness)
        self.regional_best_index = Mutable(torch.zeros(self.following_sub_swarm_size, dtype=torch.int, device=device))
        self.global_best_location = Mutable(torch.zeros(self.dim, device=device))
        self.global_best_fitness = Mutable(torch.tensor(torch.inf, device=device))

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
        if self.iteration < 0.9 * self.max_iteration:
            self._update_strategy_1(fitness)
        else:
            self._update_strategy_2(fitness)
        self.iteration += 1

    @trace_impl(step)
    def trace_step(self):
        fitness = self.evaluate(self.population)
        cond = self.iteration < 0.9 * self.max_iteration
        _if_else_ = TracingCond(self._update_strategy_1, self._update_strategy_2)
        _if_else_.cond(cond, fitness)
        self.iteration += 1

    def _update_strategy_1(self, fitness: torch.Tensor):
        self._cond_regroup(fitness)
        # Update personal_best
        compare = self.personal_best_fitness > fitness
        personal_best_location = torch.where(compare[:, None], self.population, self.personal_best_location)
        personal_best_fitness = torch.where(compare, fitness, self.personal_best_fitness)
        # Update dynamic swarms
        dynamic_size = self.dynamic_sub_swarm_size * self.dynamic_sub_swarms_num
        dynamic_size_tuple = (self.dynamic_sub_swarms_num, self.dynamic_sub_swarm_size)
        dynamic_swarms_location = self.population[:dynamic_size, :].view(*dynamic_size_tuple, self.dim)
        dynamic_swarms_fitness = fitness[:dynamic_size].view(*dynamic_size_tuple)
        dynamic_swarms_velocity = self.velocity[:dynamic_size, :].view(*dynamic_size_tuple, self.dim)
        dynamic_swarms_pbest = personal_best_location[:dynamic_size, :].view(*dynamic_size_tuple, self.dim)
        # Update following swarm
        following_swarm_location = self.population[dynamic_size:, :]
        following_swarm_velocity = self.velocity[dynamic_size:, :]
        following_swarm_pbest = personal_best_location[dynamic_size:, :]
        # Update local_best
        local_best_fitness, local_best_index = torch.min(dynamic_swarms_fitness, dim=1)  # shape:(dynamic_sub_swarms_num,)
        local_best_location = torch.index_select(dynamic_swarms_location, 1, local_best_index).diagonal().T
        # Update regional_best
        regional_best_location = self.population[self.regional_best_index, :]
        # Calculate Dynamic Swarms Velocity
        rand_pbest = torch.rand(self.pop_size, self.dim, device=self.population.device)
        rand_lbest = torch.rand(
            self.dynamic_sub_swarms_num,
            self.dynamic_sub_swarm_size,
            self.dim,
            device=self.population.device,
        )
        rand_rbest = torch.rand(self.following_sub_swarm_size, self.dim, device=self.population.device)
        dynamic_swarms_rand_pbest = rand_pbest[:dynamic_size, :].view(*dynamic_size_tuple, self.dim)
        dynamic_swarms_velocity = (
            self.w * dynamic_swarms_velocity
            + self.c_pbest * dynamic_swarms_rand_pbest * (dynamic_swarms_pbest - dynamic_swarms_location)
            + self.c_lbest * rand_lbest * (local_best_location[:, None, :] - dynamic_swarms_location)
        )
        # Calculate Following Swarm Velocity
        following_swarm_rand_pbest = rand_pbest[dynamic_size:, :]
        following_swarm_velocity = (
            self.w * following_swarm_velocity
            + self.c_pbest * following_swarm_rand_pbest * (following_swarm_pbest - following_swarm_location)
            + self.c_rbest * rand_rbest * (regional_best_location - following_swarm_location)
        )
        # Update Population
        dynamic_swarms_velocity = dynamic_swarms_velocity.view(dynamic_size, self.dim)
        velocity = torch.cat([dynamic_swarms_velocity, following_swarm_velocity], dim=0)
        population = self.population + velocity
        self.population = clamp(population, self.lb, self.ub)
        self.velocity = clamp(velocity, self.lb, self.ub)
        self.personal_best_location = personal_best_location
        self.personal_best_fitness = personal_best_fitness
        self.local_best_location = local_best_location
        self.local_best_fitness = local_best_fitness

    def _cond_regroup(self, fitness: torch.Tensor):
        if (self.iteration % self.regrouped_iteration_num) == 0:
            self._regroup(fitness)

    @trace_impl(_cond_regroup)
    def _trace_cond_regroup(self, fitness: torch.Tensor):
        _if_else_regroup_ = TracingCond(self._regroup, lambda _: None)
        _if_else_regroup_.cond((self.iteration % self.regrouped_iteration_num) == 0, fitness)

    def _regroup(self, fitness: torch.Tensor):
        sort_index = torch.argsort(fitness, dim=0)
        dynamic_size = self.dynamic_sub_swarm_size * self.dynamic_sub_swarms_num
        dynamic_swarm_population_index = sort_index[:dynamic_size]
        dynamic_swarm_population_index = torch.randperm(dynamic_size, device=self.population.device)
        regroup_index = torch.cat([dynamic_swarm_population_index, sort_index[dynamic_size:]])

        population = self.population[regroup_index]
        velocity = self.velocity[regroup_index]
        personal_best_location = self.personal_best_location[regroup_index]
        personal_best_fitness = self.personal_best_fitness[regroup_index]

        dynamic_swarm_fitness = fitness[:dynamic_size]
        regional_best_index = torch.argsort(dynamic_swarm_fitness, dim=0)[: self.following_sub_swarm_size]

        self.population = population
        self.velocity = velocity
        self.personal_best_location = personal_best_location
        self.personal_best_fitness = personal_best_fitness
        self.regional_best_index = regional_best_index

    def _update_strategy_2(self, fitness: torch.Tensor):
        # Update personal_best
        compare = self.personal_best_fitness > fitness
        personal_best_location = torch.where(compare[:, None], self.population, self.personal_best_location)
        personal_best_fitness = torch.where(compare, fitness, self.personal_best_fitness)
        # Update global_best
        global_best_fitness, global_best_idx = torch.min(personal_best_fitness, dim=0)
        global_best_location = personal_best_location[global_best_idx]
        rand_pbest = torch.rand(self.pop_size, self.dim, device=self.population.device)
        rand_gbest = torch.rand(self.pop_size, self.dim, device=self.population.device)
        velocity = (
            self.w * self.velocity
            + self.c_pbest * rand_pbest * (personal_best_location - self.population)
            + self.c_gbest * rand_gbest * (global_best_location - self.population)
        )
        population = self.population + velocity
        # Update population
        self.population = clamp(population, self.lb, self.ub)
        self.velocity = clamp(velocity, self.lb, self.ub)
        self.personal_best_location = personal_best_location
        self.personal_best_fitness = personal_best_fitness
        self.global_best_location = global_best_location
        self.global_best_fitness = global_best_fitness

import torch

from evox.core import Algorithm, Mutable, Parameter
from evox.operators.crossover import (
    DE_arithmetic_recombination,
    DE_binary_crossover,
    DE_differential_sum,
    DE_exponential_crossover,
)
from evox.operators.selection import select_rand_pbest
from evox.utils import clamp

"""
Strategy codes(4 bits): [base_vec_prim, base_vec_sec, diff_num, cross_strategy]
base_vec      : 0="rand", 1="best", 2="pbest", 3="current"
cross_strategy: 0=bin   , 1=exp   , 2=arith
"""

rand_1_bin = [0, 0, 1, 0]
rand_2_bin = [0, 0, 2, 0]
current2rand_1 = [0, 0, 1, 2]  # current2rand_1 <==> rand_1_arith
rand2best_2_bin = [0, 1, 2, 0]
current2pbest_1_bin = [3, 2, 1, 0]


class CoDE(Algorithm):
    """The implementation of CoDE algorithm.

    Reference:
    Wang Y, Cai Z, Zhang Q. Differential evolution with composite trial vector generation strategies and control parameters[J]. IEEE transactions on evolutionary computation, 2011, 15(1): 55-66.
    """

    def __init__(
        self,
        pop_size: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        diff_padding_num: int = 5,
        param_pool: torch.Tensor = torch.tensor([[1, 0.1], [1, 0.9], [0.8, 0.2]]),
        replace: bool = False,
        device: torch.device | None = None,
    ):
        """
        Initialize the CoDE algorithm with the given parameters.

        :param pop_size: The size of the population.
        :param lb: The lower bounds of the search space. Must be a 1D tensor.
        :param ub: The upper bounds of the search space. Must be a 1D tensor.
        :param diff_padding_num: The number of differential padding vectors to use. Defaults to 5.
        :param param_pool: A tensor of control parameter pairs for the algorithm. Defaults to a predefined tensor.
        :param replace: A boolean indicating whether to replace individuals in the population. Defaults to False.
        :param device: The device to use for tensor computations. Defaults to None.
        """
        super().__init__()
        device = torch.get_default_device() if device is None else device
        dim = lb.shape[0]
        # parameters
        self.param_pool = Parameter(param_pool, device=device)
        # set value
        lb = lb[None, :].to(device=device)
        ub = ub[None, :].to(device=device)
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.replace = replace
        self.pop_size = pop_size
        self.diff_padding_num = diff_padding_num
        self.strategies = torch.tensor([rand_1_bin, rand_2_bin, current2rand_1], device=device)
        # setup
        self.best_index = Mutable(torch.tensor(0, device=device))
        self.pop = Mutable(torch.randn(pop_size, dim, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((self.pop_size,), fill_value=torch.inf, device=device))

    def step(self):
        """Perform one iteration of the CoDE algorithm.

        This step is composed of the following steps:
        1. Generate trial vectors using the differential sum.
        2. Apply crossover to generate a new vector.
        3. Apply mutation to generate a new vector.
        4. Update the population and fitness values.
        """
        device = self.pop.device
        indices = torch.arange(self.pop_size, device=device)

        param_ids = torch.randint(0, 3, (3, self.pop_size), device=device)

        base_vec_prim_type = self.strategies[:, 0]
        base_vec_sec_type = self.strategies[:, 1]
        num_diff_vectors = self.strategies[:, 2]
        cross_strategy = self.strategies[:, 3]

        params = self.param_pool[param_ids]
        differential_weight = params[:, :, 0]
        cross_probability = params[:, :, 1]

        trial_vectors = torch.zeros((3, self.pop_size, self.dim), device=device)

        for i in range(3):
            difference_sum, rand_vec_idx = DE_differential_sum(
                self.diff_padding_num,
                num_diff_vectors[i],
                indices,
                self.pop,
                # self.replace
            )

            rand_vec = self.pop[rand_vec_idx]
            best_vec = torch.tile(self.pop[self.best_index].unsqueeze(0), (self.pop_size, 1))
            pbest_vec = select_rand_pbest(0.05, self.pop, self.fit)
            current_vec = self.pop[indices]

            vec_merge = torch.stack((rand_vec, best_vec, pbest_vec, current_vec))
            base_vec_prim = vec_merge[base_vec_prim_type[i]]
            base_vec_sec = vec_merge[base_vec_sec_type[i]]

            base_vec = base_vec_prim + differential_weight[i].unsqueeze(1) * (base_vec_sec - base_vec_prim)
            mutation_vec = base_vec + difference_sum * differential_weight[i].unsqueeze(1)

            trial_vec = torch.zeros(self.pop_size, self.dim, device=device)
            trial_vec = torch.where(
                cross_strategy[i] == 0, DE_binary_crossover(mutation_vec, current_vec, cross_probability[i]), trial_vec
            )
            trial_vec = torch.where(
                cross_strategy[i] == 1, DE_exponential_crossover(mutation_vec, current_vec, cross_probability[i]), trial_vec
            )
            trial_vec = torch.where(
                cross_strategy[i] == 2, DE_arithmetic_recombination(mutation_vec, current_vec, cross_probability[i]), trial_vec
            )
            trial_vectors = torch.where(
                (torch.arange(3, device=device) == i).unsqueeze(1).unsqueeze(2), trial_vec.unsqueeze(0), trial_vectors
            )

        trial_vectors = clamp(trial_vectors.reshape(3 * self.pop_size, self.dim), self.lb, self.ub)
        trial_fitness = self.evaluate(trial_vectors)

        indices = torch.arange(3 * self.pop_size, device=device).reshape(3, self.pop_size)
        trans_fit = trial_fitness[indices]

        min_indices = torch.argmin(trans_fit, dim=0)
        min_indices_global = indices[min_indices, torch.arange(self.pop_size, device=device)]

        trial_fitness_select = trial_fitness[min_indices_global]
        trial_vectors_select = trial_vectors[min_indices_global]

        compare = trial_fitness_select <= self.fit

        self.pop = torch.where(compare[:, None], trial_vectors_select, self.pop)
        self.fit = torch.where(compare, trial_fitness_select, self.fit)
        self.best_index = torch.argmin(self.fit)

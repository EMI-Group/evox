import torch

from evox.core import Algorithm, Mutable, Parameter
from evox.operators.crossover import (
    DE_binary_crossover,
    DE_differential_sum,
)
from evox.operators.selection import select_rand_pbest
from evox.utils import clamp


class SHADE(Algorithm):
    """The implementation of SHADE algorithm.

    Reference:
    Tanabe R, Fukunaga A.
    Success-history based parameter adaptation for differential evolution[C]//2013
    IEEE congress on evolutionary computation. IEEE, 2013: 71-78.
    """

    def __init__(
        self,
        pop_size: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        diff_padding_num: int = 9,
        device: torch.device | None = None,
    ):
        """
        Initialize the SHADE algorithm with the given parameters.

        :param pop_size: The size of the population.
        :param lb: The lower bounds of the search space. Must be a 1D tensor.
        :param ub: The upper bounds of the search space. Must be a 1D tensor.
        :param diff_padding_num: The number of differential padding vectors to use. Defaults to 9.
        :param device: The device to use for tensor computations (e.g., "cpu" or "cuda"). Defaults to None.
        """
        super().__init__()
        device = torch.get_default_device() if device is None else device
        assert pop_size >= 9
        assert lb.shape == ub.shape and lb.ndim == 1 and ub.ndim == 1 and lb.dtype == ub.dtype
        dim = lb.shape[0]
        # parameters
        self.num_diff_vectors = Parameter(torch.tensor(1), device=device)
        # set value
        lb = lb[None, :].to(device=device)
        ub = ub[None, :].to(device=device)
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.pop_size = pop_size
        self.diff_padding_num = diff_padding_num
        # setup
        self.best_index = Mutable(torch.tensor(0, device=device))
        self.Memory_FCR = Mutable(torch.full((2, pop_size), fill_value=0.5, device=device))
        self.pop = Mutable(torch.randn(pop_size, dim, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((pop_size,), fill_value=torch.inf, device=device))

    def step(self):
        """
        Perform a single step of the SHADE algorithm.

        This involves the following sub-steps:
        1. Generate trial vectors using the SHADE algorithm.
        2. Evaluate the fitness of the trial vectors.
        3. Update the population.
        4. Update the memory.
        """
        device = self.pop.device
        indices = torch.arange(self.pop_size, device=device)

        FCR_ids = torch.randperm(self.pop_size)
        M_F_vect = self.Memory_FCR[0, FCR_ids]
        M_CR_vect = self.Memory_FCR[1, FCR_ids]

        F_vect = torch.randn(self.pop_size, device=device) * 0.1 + M_F_vect
        F_vect = clamp(F_vect, torch.zeros(self.pop_size, device=device), torch.ones(self.pop_size, device=device))

        CR_vect = torch.randn(self.pop_size, device=device) * 0.1 + M_CR_vect
        CR_vect = clamp(CR_vect, torch.zeros(self.pop_size, device=device), torch.ones(self.pop_size, device=device))

        difference_sum, rand_vect_idx = DE_differential_sum(
            self.diff_padding_num, torch.tile(self.num_diff_vectors, (self.pop_size,)), indices, self.pop
        )
        pbest_vect = select_rand_pbest(0.05, self.pop, self.fit)
        current_vect = self.pop[indices]

        base_vector_prim = current_vect
        base_vector_sec = pbest_vect

        base_vector = base_vector_prim + F_vect.unsqueeze(1) * (base_vector_sec - base_vector_prim)
        mutation_vector = base_vector + difference_sum * F_vect.unsqueeze(1)

        trial_vector = DE_binary_crossover(mutation_vector, current_vect, CR_vect)
        trial_vector = clamp(trial_vector, self.lb, self.ub)

        trial_fitness = self.evaluate(trial_vector)

        compare = trial_fitness < self.fit

        population_update = torch.where(compare[:, None], trial_vector, self.pop)
        fitness_update = torch.where(compare, trial_fitness, self.fit)

        self.pop = population_update
        self.fit = fitness_update

        self.best_index = torch.argmin(self.fit)

        S_F = torch.full((self.pop_size,), fill_value=torch.nan, device=device)
        S_CR = torch.full((self.pop_size,), fill_value=torch.nan, device=device)
        S_delta = torch.full((self.pop_size,), fill_value=torch.nan, device=device)

        deltas = self.fit - trial_fitness

        for i in range(self.pop_size):  #  get_success_delta
            is_success = compare[i].float()
            F = F_vect[i]
            CR = CR_vect[i]
            delta = deltas[i]

            S_F_update_temp = torch.roll(S_F, shifts=1)
            S_F_update = torch.cat([F.unsqueeze(0), S_F_update_temp[1:]], dim=0)

            S_CR_update_temp = torch.roll(S_CR, shifts=1)
            S_CR_update = torch.cat([CR.unsqueeze(0), S_CR_update_temp[1:]], dim=0)

            S_delta_update_temp = torch.roll(S_delta, shifts=1)
            S_delta_update = torch.cat([delta.unsqueeze(0), S_delta_update_temp[1:]], dim=0)

            S_F = is_success * S_F_update + (1.0 - is_success) * S_F_update_temp
            S_CR = is_success * S_CR_update + (1.0 - is_success) * S_CR_update_temp
            S_delta = is_success * S_delta_update + (1.0 - is_success) * S_delta_update_temp

        norm_delta = S_delta / torch.nansum(S_delta)
        M_CR = torch.nansum(norm_delta * S_CR)
        M_F = torch.nansum(norm_delta * (S_F**2)) / torch.nansum(norm_delta * S_F)

        Memory_FCR_update = torch.roll(self.Memory_FCR, shifts=1, dims=1)
        Memory_FCR_update[0, 0] = M_F
        Memory_FCR_update[1, 0] = M_CR

        is_F_nan = torch.isnan(M_F)
        Memory_FCR_update = torch.where(is_F_nan, self.Memory_FCR, Memory_FCR_update)

        is_S_nan = torch.all(torch.isnan(compare))
        Memory_FCR = torch.where(is_S_nan, self.Memory_FCR, Memory_FCR_update)

        self.Memory_FCR = Memory_FCR

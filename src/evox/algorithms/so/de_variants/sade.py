import torch

from evox.core import Algorithm, Mutable
from evox.operators.crossover import (
    DE_arithmetic_recombination,
    DE_binary_crossover,
    DE_differential_sum,
    DE_exponential_crossover,
)
from evox.operators.selection import select_rand_pbest
from evox.utils import clamp

# Strategy codes(4 bits): [base_vec_prim, base_vec_sec, diff_num, cross_strategy]
# base_vec: 0="rand", 1="best", 2="pbest", 3="current", cross_strategy: 0=bin, 1=exp, 2=arith
rand_1_bin = [0, 0, 1, 0]
rand_2_bin = [0, 0, 2, 0]
rand2best_2_bin = [0, 1, 2, 0]
current2rand_1 = [0, 0, 1, 2]  # current2rand_1 <==> rand_1_arith


class SaDE(Algorithm):
    """The implementation of SaDE algorithm.

    Reference:
    Qin A K, Huang V L, Suganthan P N.
    Differential evolution algorithm with strategy adaptation for global numerical optimization[J].
    IEEE transactions on Evolutionary Computation, 2008, 13(2): 398-417.
    """

    def __init__(
        self,
        pop_size: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        diff_padding_num: int = 9,
        LP: int = 50,
        device: torch.device | None = None,
    ):
        """
        Initialize the SaDE algorithm with the given parameters.

        :param pop_size: The size of the population.
        :param lb: The lower bounds of the search space. Must be a 1D tensor.
        :param ub: The upper bounds of the search space. Must be a 1D tensor.
        :param diff_padding_num: The number of differential padding vectors to use. Defaults to 9.
        :param LP: The size of memory. Defaults to 50.
        :param device: The device to use for tensor computations (e.g., "cpu" or "cuda"). Defaults to None.
        """
        super().__init__()
        device = torch.get_default_device() if device is None else device
        assert pop_size >= 9
        assert lb.shape == ub.shape and lb.ndim == 1 and ub.ndim == 1 and lb.dtype == ub.dtype
        dim = lb.shape[0]
        # parameters
        # set value
        lb = lb[None, :].to(device=device)
        ub = ub[None, :].to(device=device)
        self.lb = lb
        self.ub = ub
        self.LP = LP
        self.dim = dim
        self.pop_size = pop_size
        self.diff_padding_num = diff_padding_num
        self.strategy_pool = torch.tensor([rand_1_bin, rand2best_2_bin, rand_2_bin, current2rand_1], device=device)
        # setup
        self.gen_iter = Mutable(torch.tensor(0, device=device))
        self.best_index = Mutable(torch.tensor(0, device=device))
        self.Memory_FCR = Mutable(torch.full((2, 100), fill_value=0.5, device=device))
        self.pop = Mutable(torch.randn(pop_size, dim, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((self.pop_size,), fill_value=torch.inf, device=device))
        self.success_memory = Mutable(torch.full((LP, 4), fill_value=0, device=device))
        self.failure_memory = Mutable(torch.full((LP, 4), fill_value=0, device=device))
        self.CR_memory = Mutable(torch.full((LP, 4), fill_value=torch.nan, device=device))
        # Others
        self.generator = torch.Generator(device=device)

    def _get_strategy_ids(self, strategy_p: torch.Tensor, device: torch.device):
        strategy_ids = torch.multinomial(strategy_p, self.pop_size, replacement=True, generator=self.generator)
        return strategy_ids

    def _vmap_get_strategy_ids(self, strategy_p: torch.Tensor, device: torch.device):
        # TODO: since torch.multinomial is not supported in vmap, we have to use torch.randint
        strategy_ids = torch.randint(0, 4, (self.pop_size,), device=device)
        return strategy_ids

    def step(self):
        """
        Execute a single optimization step of the SaDE algorithm.

        This involves the following sub-steps:
        1. Generate new population using differential evolution.
        2. Evaluate the fitness of the new population.
        3. Update the best individual and best fitness.
        4. Update the success and failure memory.
        5. Update the CR memory.
        """
        device = self.pop.device
        indices = torch.arange(self.pop_size, device=device)

        CRM_init = torch.tensor([0.5, 0.5, 0.5, 0.5], device=device)
        strategy_p_init = torch.tensor([0.25, 0.25, 0.25, 0.25], device=device)

        success_sum = torch.sum(self.success_memory, dim=0)
        failure_sum = torch.sum(self.failure_memory, dim=0)
        S_mat = (success_sum / (success_sum + failure_sum)) + 0.01

        strategy_p_update = S_mat / torch.sum(S_mat)
        strategy_p = torch.where(self.gen_iter >= self.LP, strategy_p_update, strategy_p_init)

        CRM_update = torch.median(self.CR_memory, dim=0)[0]
        CRM = torch.where(self.gen_iter > self.LP, CRM_update, CRM_init)

        strategy_ids = self._get_strategy_ids(strategy_p, device)

        CRs_vec = torch.randn((self.pop_size, 4), device=device) * 0.1 + CRM
        CRs_vec_repair = torch.randn((self.pop_size, 4), device=device) * 0.1 + CRM

        mask = (CRs_vec < 0) | (CRs_vec > 1)
        CRs_vec = torch.where(mask, CRs_vec_repair, CRs_vec)

        differential_weight = torch.randn(self.pop_size, device=device) * 0.3 + 0.5
        cross_probability = torch.gather(CRs_vec, 1, strategy_ids[:, None])[:, 0]

        strategy_code = self.strategy_pool[strategy_ids]
        base_vec_prim_type = strategy_code[:, 0]
        base_vec_sec_type = strategy_code[:, 1]
        num_diff_vectors = strategy_code[:, 2]
        cross_strategy = strategy_code[:, 3]

        difference_sum, rand_vec_idx = DE_differential_sum(self.diff_padding_num, num_diff_vectors, indices, self.pop)

        rand_vec = self.pop[rand_vec_idx]
        best_vec = torch.tile(self.pop[self.best_index].unsqueeze(0), (self.pop_size, 1))
        pbest_vec = select_rand_pbest(0.05, self.pop, self.fit)
        current_vec = self.pop[indices]
        vector_merge = torch.stack([rand_vec, best_vec, pbest_vec, current_vec])

        base_vector_prim = torch.zeros(self.pop_size, self.dim, device=device)
        base_vector_sec = torch.zeros(self.pop_size, self.dim, device=device)

        for i in range(4):
            base_vector_prim = torch.where(base_vec_prim_type.unsqueeze(1) == i, vector_merge[i], base_vector_prim)
            base_vector_sec = torch.where(base_vec_sec_type.unsqueeze(1) == i, vector_merge[i], base_vector_sec)

        base_vector = base_vector_prim + differential_weight.unsqueeze(1) * (base_vector_sec - base_vector_prim)
        mutation_vector = base_vector + difference_sum * differential_weight.unsqueeze(1)

        trial_vector = torch.zeros(self.pop_size, self.dim, device=device)
        trial_vector = torch.where(
            cross_strategy.unsqueeze(1) == 0,
            DE_binary_crossover(mutation_vector, current_vec, cross_probability),
            trial_vector,
        )
        trial_vector = torch.where(
            cross_strategy.unsqueeze(1) == 1,
            DE_exponential_crossover(mutation_vector, current_vec, cross_probability),
            trial_vector,
        )
        trial_vector = torch.where(
            cross_strategy.unsqueeze(1) == 2,
            DE_arithmetic_recombination(mutation_vector, current_vec, cross_probability),
            trial_vector,
        )
        trial_vector = clamp(trial_vector, self.lb, self.ub)

        CRs_vec = torch.gather(CRs_vec, 1, strategy_ids.unsqueeze(1)).squeeze(1)

        trial_fitness = self.evaluate(trial_vector)

        self.gen_iter = self.gen_iter + 1

        compare = trial_fitness <= self.fit

        self.pop = torch.where(compare[:, None], trial_vector, self.pop)
        self.fit = torch.where(compare, trial_fitness, self.fit)

        self.best_index = torch.argmin(self.fit)

        """Update memories"""
        success_memory = torch.roll(self.success_memory, 1, 0)
        success_memory[0, :] = 0
        failure_memory = torch.roll(self.failure_memory, 1, 0)
        failure_memory[0, :] = 0

        for i in range(self.pop_size):
            success_memory_up = success_memory.clone()
            success_memory_up[0][strategy_ids[i]] += 1
            success_memory = torch.where(compare[i], success_memory_up, success_memory)

            failure_memory_up = failure_memory.clone()
            failure_memory_up[0][strategy_ids[i]] += 1
            failure_memory = torch.where(compare[i], failure_memory, failure_memory_up)

        CR_memory = self.CR_memory

        for i in range(self.pop_size):
            str_idx = strategy_ids[i]

            CR_mk_up = torch.roll(CR_memory.t()[str_idx], 1)
            CR_mk_up[0] = CRs_vec[i]

            CR_memory_up = CR_memory.clone().t()
            CR_memory_up[str_idx][:] = CR_mk_up
            CR_memory_up = CR_memory_up.t()
            CR_memory = torch.where(compare[i], CR_memory_up, CR_memory)

        self.success_memory = success_memory
        self.failure_memory = failure_memory
        self.CR_memory = CR_memory

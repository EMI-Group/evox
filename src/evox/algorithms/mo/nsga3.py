from typing import Callable, Optional

import torch

from evox.core import Algorithm, Mutable, vmap
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling
from evox.operators.selection import non_dominate_rank, tournament_selection_multifit
from evox.utils import clamp


def _get_table_row_inner(bool_ref_candidate: torch.Tensor, upper_bound: torch.Tensor):
    # true_indices = torch.nonzero_static(bool_ref_candidate, size=bool_ref_candidate.size(0), fill_value=upper_bound).squeeze(-1)
    true_indices = torch.where(
        bool_ref_candidate,
        torch.arange(bool_ref_candidate.size(0), dtype=torch.int32, device=torch.get_default_device()),
        upper_bound,
    )
    true_indices = torch.sort(true_indices, dim=0).values
    return true_indices.to(torch.int32)


vmap_get_table_row = vmap(
    _get_table_row_inner,
    in_dims=(0, None),
)


def _select_from_index_by_min_inner(
    group_id: torch.Tensor,
    group_dist: torch.Tensor,
    idx: torch.Tensor,
):
    min_idx = torch.argmin(torch.where(group_id == idx.unsqueeze(0), group_dist, torch.inf)).to(torch.int32)
    return min_idx


vmap_select_from_index_by_min = vmap(
    _select_from_index_by_min_inner,
    in_dims=(None, None, 0),
)


def _get_extreme_inner(norm_fit: torch.Tensor, w: torch.Tensor):
    return torch.argmin(torch.max(norm_fit / w.unsqueeze(0), dim=1).values)


vmap_get_extreme = vmap(
    _get_extreme_inner,
    in_dims=(None, 0),
)


class NSGA3(Algorithm):
    """
    An implementation of the reference-point based many-objective NSGA-II (NSGA-III) for many-objective optimization problems.

    This class provides a framework for solving many-objective optimization problems using reference-points,
    which is widely used for many-objective optimization.

    :references:
        - "An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based Nondominated Sorting Approach, Part I: Solving Problems With Box Constraints" IEEE Transactions on Evolutionary Computation.
          `Link <https://ieeexplore.ieee.org/document/6600851>`_
    """

    def __init__(
        self,
        pop_size: int,
        n_objs: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        selection_op: Optional[Callable] = None,
        mutation_op: Optional[Callable] = None,
        crossover_op: Optional[Callable] = None,
        data_type: Optional[torch.dtype] = None,
        device: torch.device | None = None,
    ):
        """Initializes the NSGA-III algorithm.

        :param pop_size: The size of the population.
        :param n_objs: The number of objective functions in the optimization problem.
        :param lb: The lower bounds for the decision variables (1D tensor).
        :param ub: The upper bounds for the decision variables (1D tensor).
        :param selection_op: The selection operation for evolutionary strategy (optional).
        :param mutation_op: The mutation operation, defaults to `polynomial_mutation` if not provided (optional).
        :param crossover_op: The crossover operation, defaults to `simulated_binary` if not provided (optional).
        :param data_type: The data type for the decision variables (optional). Defaults to torch.float32.
        :param device: The device on which computations should run (optional). Defaults to PyTorch's default device.
        """

        super().__init__()
        self.pop_size = pop_size
        self.n_objs = n_objs
        if device is None:
            device = torch.get_default_device()
        # check
        assert lb.shape == ub.shape and lb.ndim == 1 and ub.ndim == 1
        assert lb.dtype == ub.dtype and lb.device == ub.device
        self.dim = lb.shape[0]
        # write to self
        self.lb = lb.to(device=device)
        self.ub = ub.to(device=device)

        self.selection = selection_op
        self.mutation = mutation_op
        self.crossover = crossover_op

        if self.selection is None:
            self.selection = tournament_selection_multifit
        if self.mutation is None:
            self.mutation = polynomial_mutation
        if self.crossover is None:
            self.crossover = simulated_binary

        if data_type == torch.bool:
            population = torch.rand(self.pop_size, self.dim, device=device)
            population = population > 0.5
        else:
            length = ub - lb
            population = torch.rand(self.pop_size, self.dim, device=device)
            population = length * population + lb

        self.pop = Mutable(population)
        self.fit = Mutable(torch.full((self.pop_size, self.n_objs), torch.inf, device=device))
        self.rank = Mutable(torch.full((self.pop_size,), torch.inf, device=device))
        self.ref = uniform_sampling(self.pop_size, self.n_objs)[0]

    def init_step(self):
        """
        Perform the initialization step of the workflow.

        Calls the `init_step` of the algorithm if overwritten; otherwise, its `step` method will be invoked.
        """
        self.fit = self.evaluate(self.pop)
        self.rank = non_dominate_rank(self.fit)

    def step(self):
        """Perform the optimization step of the workflow."""
        mating_pool = self.selection(self.pop_size, [self.rank])
        crossovered = self.crossover(self.pop[mating_pool])
        offspring = self.mutation(crossovered, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)
        off_fit = self.evaluate(offspring)
        merge_pop = torch.cat([self.pop, offspring], dim=0)
        merge_fit = torch.cat([self.fit, off_fit], dim=0)
        shuffled_idx = torch.randperm(merge_pop.shape[0])
        merge_pop = merge_pop[shuffled_idx]
        merge_fit = merge_fit[shuffled_idx]
        rank = non_dominate_rank(merge_fit)
        worst_rank = torch.topk(rank, self.pop_size + 1, largest=False)[0][-1]
        candi_idx = torch.where(rank <= worst_rank)[0]
        merge_pop = merge_pop[candi_idx]
        merge_fit = merge_fit[candi_idx]
        rank = rank[candi_idx]
        device = self.pop.device
        # Normalize
        ideal_point = torch.min(merge_fit, dim=0)[0]
        norm_fit = merge_fit - ideal_point
        weight = torch.eye(self.n_objs, device=device) + 1e-6
        ex_idx = vmap_get_extreme(norm_fit, weight)
        extreme = norm_fit[ex_idx]  # shape: (n_objs, dim)
        if torch.linalg.matrix_rank(extreme) == self.n_objs:
            hyperplane = torch.linalg.solve(extreme, torch.ones(self.n_objs, device=device))
            intercepts = 1.0 / hyperplane
        else:
            intercepts = torch.max(norm_fit, dim=0).values
        norm_fit = norm_fit / intercepts.unsqueeze(0)
        shuffled_idx = torch.randperm(self.ref.shape[0])
        ref = self.ref[shuffled_idx]
        # Calculate distances by cosine similarity
        distances = self._compute_distances(norm_fit, ref)
        # Associate each solution with its nearest reference point
        group_dist, group_id = torch.min(distances, dim=1)
        # count the number of individuals for each group id
        selected_group_id = group_id[rank < worst_rank]
        rho = torch.bincount(selected_group_id, minlength=ref.shape[0]).to(torch.int32)
        selected_num = torch.sum(rho, dtype=torch.int32)
        candi_group_id = group_id[rank == worst_rank]
        rho_last = torch.bincount(candi_group_id, minlength=ref.shape[0]).to(torch.int32)
        upper_bound = torch.tensor(
            merge_pop.shape[0] + merge_pop.shape[1] + merge_fit.shape[1] + 1, dtype=torch.int32, device=device
        )
        rho = torch.where(rho_last == 0, upper_bound, rho)
        group_id = torch.where(rank == worst_rank, group_id, upper_bound).to(torch.int32)
        row_indices = torch.arange(ref.shape[0], device=device).to(torch.int32)

        # first selection stage
        rho_level = 0
        _selected_ref = rho == rho_level
        selected_ref = torch.where(_selected_ref, row_indices, upper_bound)
        candi_idx = vmap_select_from_index_by_min(group_id, group_dist, selected_ref)
        rank[candi_idx[_selected_ref]] = worst_rank - 1
        rho_last = torch.where(_selected_ref, rho_last - 1, rho_last)
        rho = torch.where(_selected_ref, rho_level + 1, rho)
        rho = torch.where(rho_last == 0, upper_bound, rho)
        selected_num += torch.sum(_selected_ref)

        # second selection stage
        group_id[candi_idx[_selected_ref]] = upper_bound
        bool_ref_candidates = row_indices[:, None] == group_id[None, :]  # shape (ref.shape[0], group_id.shape[0])
        ref_candidates = vmap_get_table_row(bool_ref_candidates, upper_bound)
        ref_cand_idx = torch.zeros_like(rho)
        while selected_num < self.pop_size:
            rho_level = torch.min(rho)
            _selected_ref = rho == rho_level
            candi_idx = ref_candidates[row_indices, ref_cand_idx]
            rank[candi_idx[_selected_ref]] = worst_rank - 1
            ref_cand_idx = torch.where(_selected_ref, ref_cand_idx + 1, ref_cand_idx)
            rho_last = torch.where(_selected_ref, rho_last - 1, rho_last)
            rho = torch.where(_selected_ref, rho_level + 1, rho)
            rho = torch.where(rho_last == 0, upper_bound, rho)
            selected_num += torch.sum(_selected_ref)

        # truncate to pop_size
        dif = selected_num - self.pop_size
        candi_idx = torch.where(_selected_ref, candi_idx, upper_bound)
        sorted_index = torch.sort(candi_idx, stable=False)[0]
        rank[sorted_index[:dif]] = worst_rank

        # get final pop and fit
        self.pop = merge_pop[rank < worst_rank]
        self.fit = merge_fit[rank < worst_rank]
        self.rank = rank[rank < worst_rank]

    def _get_extreme(self, norm_fit: torch.Tensor, w: torch.Tensor):
        return torch.argmin(torch.max(norm_fit / w.unsqueeze(0), dim=1).values)

    def _compute_distances(self, fit: torch.Tensor, ref: torch.Tensor):
        # Normalize solutions and reference points to unit vectors
        fit_magnitude = torch.norm(fit, dim=1, keepdim=True).clamp_min(1e-10)  # Shape: (pop_size, 1)
        fit_norm = fit / fit_magnitude
        ref_norm = ref / torch.norm(ref, dim=1, keepdim=True).clamp_min(1e-10)

        # Compute cosine similarity (dot product of normalized vectors)
        cosine_sim = torch.matmul(fit_norm, ref_norm.T)  # Shape: (pop_size, ref_size)

        # Compute the angular distance component (sqrt(1 - cosine_similarity^2))
        angular_distance = torch.sqrt((1 - cosine_sim**2).clamp_min(1e-10))  # Shape: (pop_size, ref_size)

        # Compute the final distance by multiplying magnitude with angular distance
        distances = fit_magnitude * angular_distance  # Shape: (pop_size, ref_size)
        return distances

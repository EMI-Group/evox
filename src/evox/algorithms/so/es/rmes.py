# --------------------------------------------------------------------------------------
# 1. This code implements algorithms described in the following papers:
#
# Title: A Simple Yet Efficient Evolution Strategy for Large-Scale Black-Box Optimization (RMES)
# Link: https://ieeexplore.ieee.org/document/8080257df（RMES)
#
# 2. This code has been inspired by or utilizes the algorithmic implementation from evosax.
#
# More information about evosax can be found at the following URL:
# GitHub Link: https://github.com/RobertTLange/evosax
# --------------------------------------------------------------------------------------

import jax
import jax.numpy as jnp
from jax import lax
import evox
from .cma_es import CMAES
from .sort_utils import sort_by_key


@evox.jit_class
class RMES(CMAES):
    def __init__(
        self,
        center_init,
        init_stdev,
        pop_size=None,
        recombination_weights=None,
        cm=1,
        elite_ratio=0.5,
        memory_size=10,
        mean_decay=0,
        sparse_threshold=0.01,
        t_uncorr=10,
    ):
        super().__init__(center_init, init_stdev, pop_size, recombination_weights, cm)
        self.elite_ratio = elite_ratio
        self.elite_popsize = max(1, int(self.pop_size * self.elite_ratio))
        self.memory_size = memory_size
        self.mean_decay = mean_decay
        self.sparse_threshold = sparse_threshold
        self.c_sigma = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.t_uncorr = t_uncorr
        self.q_star = 0.5
        self.c_s = 0.3
        self.d_sigma = 1.0
        self.s_rank_rate = 0.0

    def setup(self, key):
        base_state = super().setup(key)
        P = jnp.zeros((self.dim, self.memory_size))
        p_sigma = jnp.zeros(self.dim)
        t_gap = jnp.zeros(self.memory_size)
        s_rank_rate = 0.0
        fitness_archive = jnp.zeros(self.pop_size) + 1e20
        return base_state.update(
            P=P,
            t_gap=t_gap,
            s_rank_rate=s_rank_rate,
            fitness_archive=fitness_archive,
            p_sigma=p_sigma,
        )

    def tell(self, state, fitness):
        # Base CMA-ES updates

        fitness, population = sort_by_key(fitness, state.population)

        mean = self._update_mean(state.mean, population)
        delta_mean = mean - state.mean

        ps = self._update_ps(state.ps, state.invsqrtC, state.sigma, delta_mean)

        hsig = (
            jnp.linalg.norm(ps) / jnp.sqrt(1 - (1 - self.cs) ** (2 * state.count_iter))
            < (1.4 + 2 / (self.dim + 1)) * self.chiN
        )
        pc = self._update_pc(state.pc, ps, delta_mean, state.sigma, hsig)
        C = self._update_C(state.C, pc, state.sigma, population, state.mean, hsig)
        sigma = self._update_sigma(state.sigma, ps)
        s_rank_rate = self._rank_success_rule(fitness, state.fitness_archive)
        B, D, invsqrtC = lax.cond(
            state.count_iter % self.decomp_per_iter == 0,
            self._decomposition_C,
            lambda _C: (state.B, state.D, state.invsqrtC),
            C,
        )

        new_state = state.update(
            mean=mean, ps=ps, pc=pc, C=C, sigma=sigma, B=B, D=D, invsqrtC=invsqrtC
        )
        # RmES specific updates
        p_sigma_new = (1 - self.c_sigma) * state.p_sigma + jnp.sqrt(
            self.c_sigma * (2 - self.c_sigma) * self.mueff
        ) * (mean - state.mean) / state.sigma
        P, t_gap = self._update_P_matrix(
            state.P, state.p_sigma, state.t_gap, state.count_iter
        )
        s_rank_rate = self._rank_success_rule(fitness, state.fitness_archive)

        return new_state.update(
            mean=mean,
            p_sigma=p_sigma_new,
            P=P,
            t_gap=t_gap,
            s_rank_rate=s_rank_rate,
            fitness_archive=fitness,
        )

    def _update_mean(self, mean, population):
        sorted_population = population[population[:, 0].argsort()]
        elite_solutions = sorted_population[: self.mu, 0:]
        weighted_sum_elite = elite_solutions.T @ self.weights
        new_mean = (1 - self.mean_decay) * mean + self.mean_decay * weighted_sum_elite

        return new_mean

    def _update_C(self, C, pc, sigma, population, old_mean, hsig):
        # Using the sparse plus low rank model for updating C
        y = (population[: self.mu] - old_mean) / sigma
        C_update = self._sparse_plus_low_rank_update(y, pc)

        C_new = (
            (1 - self.c1 - self.cmu) * C
            + self.c1 * (C_update + (1 - hsig) * self.cc * (2 - self.cc) * C)
            + self.cmu * (y.T * self.weights) @ y
        )
        return C_new

    def _update_P_matrix(self, P, p_sigma, t_gap, gen_counter):
        memory_size = P.shape[1]
        T_min = jnp.min(t_gap[1:] - t_gap[:-1])
        replace_crit = T_min > self.t_uncorr
        fill_up_crit = gen_counter < memory_size
        push_replace = jnp.logical_or(replace_crit, fill_up_crit)
        P_c1 = P.at[:, :-1].set(P[:, 1:])
        t_gap_c1 = t_gap.at[:-1].set(t_gap[1:])
        i_min = jnp.argmin(t_gap[:-1] - t_gap[1:])

        def body_fun(i, vals):
            P_c2, t_gap_c2 = vals
            replace_bool = i >= i_min

            P_c2_new = jax.lax.cond(
                replace_bool,
                lambda _: P_c2.at[:, i].set(P_c2[:, i + 1]),
                lambda _: P_c2,
                None,
            )
            t_gap_c2_new = jax.lax.cond(
                replace_bool,
                lambda _: t_gap_c2.at[i].set(t_gap_c2[i + 1]),
                lambda _: t_gap_c2,
                None,
            )

            return P_c2_new, t_gap_c2_new

        P_c2, t_gap_c2 = lax.fori_loop(0, memory_size - 1, body_fun, (P, t_gap))

        P1 = jax.lax.cond(push_replace, lambda _: P_c1, lambda _: P_c2, None)
        t_gap1 = jax.lax.cond(
            push_replace, lambda _: t_gap_c1, lambda _: t_gap_c2, None
        )

        P_new = P1.at[:, memory_size - 1].set(p_sigma)
        t_gap_new = t_gap1.at[memory_size - 1].set(gen_counter)
        return P_new, t_gap_new

    def _sparse_plus_low_rank_update(self, y, pc):
        # Aggregate along the first dimension of y to get a shape of (114,)
        y_aggregated = jnp.mean(y, axis=0)

        # Sparse update
        sparse_update = jnp.where(
            jnp.abs(jnp.outer(y_aggregated, pc)) > self.sparse_threshold,
            jnp.outer(y_aggregated, pc),
            0,
        )

        # Low rank update
        low_rank_update = jnp.outer(pc, pc)

        return sparse_update + low_rank_update

    def _rank_success_rule(self, fitness, fitness_archive):
        elite_popsize = self.weights.shape[0]
        popsize = fitness.shape[0]
        concat_all = jnp.vstack(
            [jnp.expand_dims(fitness, 1), jnp.expand_dims(fitness_archive, 1)]
        )
        ranks = jnp.zeros(concat_all.shape[0])
        ranks = ranks.at[concat_all[:, 0].argsort()].set(jnp.arange(2 * popsize))
        ranks_current = ranks[:popsize]
        ranks_current = ranks_current[ranks_current.argsort()][:elite_popsize]
        ranks_last = ranks[popsize:]
        ranks_last = ranks_last[ranks_last.argsort()][:elite_popsize]
        q = 1 / elite_popsize * jnp.sum(self.weights * (ranks_last - ranks_current))
        new_s_rank_rate = (1 - self.c_s) * self.s_rank_rate + self.c_s * (
            q - self.q_star
        )
        return new_s_rank_rate

from functools import partial
from typing import List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import vmap
from jax.tree_util import tree_map

from evox import Algorithm, State, dataclass, pytree_field, jit_class, use_state


@jit_class
@dataclass
class VectorizedCoevolution(Algorithm):
    """A container for vectorized co-evolutionary algorithms.

    Parameters
    ----------
    base_algorithms:
        A batch of base algorithms.
        Usually created from `Stateful.stack(...)`, where ... is a list of algorithms.
    dim:
        The dimension of the problem in full.
    num_subpops:
        The number of subpopulations.
    random_subpop:
        Whether to shuffle the problem dimension before co-evolution.
        When set to False, each subpopulation will corespond to a contiguous block of the decision variables,
        for example, dimension 0~9 for subpopulation 0, 10~19 for subpopulation 1, etc.
        When set to True, the decision variables will be shuffled.
    """
    base_algorithms: Algorithm = pytree_field(stack=True)
    dim: int = pytree_field(static=True)
    num_subpops: int = pytree_field(static=True)
    random_subpop: bool = pytree_field(static=True)
    dtype: jnp.dtype = pytree_field(static=True, default=jnp.float32)

    def setup(self, key: jax.Array) -> State:
        if self.random_subpop:
            key, subkey = jax.random.split(key)
            permutation = jax.random.permutation(subkey, self.dim)
        else:
            permutation = None

        best_dec = jnp.empty((self.dim,), dtype=self.dtype)
        best_fit = jnp.empty((self.num_subpops,), dtype=self.dtype)

        return State(
            coop_pops=None,
            best_dec=best_dec,
            best_fit=best_fit,
            permutation=permutation,
        )

    def init_ask(self, state: State):
        init_subpops, state = use_state(vmap(self.base_algorithms.__class__.init_ask))(
            self.base_algorithms, state
        )
        # init_subpops (num_subpops, subpop_size, sub_dim)
        init_pop = init_subpops.transpose((1, 0, 2)).reshape(-1, self.dim)

        if self.random_subpop:
            init_pop = init_pop.at[:, state.permutation].set(init_pop)

        return init_pop, state.update(coop_pops=init_pop)

    def ask(self, state: State) -> Tuple[jax.Array, State]:
        subpop, state = use_state(vmap(self.base_algorithms.__class__.ask))(
            self.base_algorithms, state
        )

        _num_subpops, pop_size, sub_dim = subpop.shape
        # co-operate
        tiled_best_dec = jnp.tile(state.best_dec, (pop_size, 1))
        coop_pops = vmap(
            lambda index: jax.lax.dynamic_update_slice(
                tiled_best_dec, subpop[index], (0, index * sub_dim)
            )
        )(jnp.arange(self.num_subpops))
        coop_pops = coop_pops.reshape((self.num_subpops * pop_size, self.dim))

        # if random_subpop is set, do a inverse permutation here.
        if self.random_subpop:
            coop_pops = coop_pops.at[:, state.permutation].set(coop_pops)

        return coop_pops, state.update(coop_pops=coop_pops)

    def init_tell(self, state, fitness):
        best_fit = jnp.min(fitness)
        state = use_state(
            vmap(self.base_algorithms.__class__.init_tell, in_axes=(0, 0, None))
        )(self.base_algorithms, state, fitness)
        best_dec = state.coop_pops[jnp.argmin(fitness)]
        if self.random_subpop:
            best_dec = best_dec[state.permutation]

        return state.update(
            best_fit=jnp.tile(best_fit, self.num_subpops),
            best_dec=best_dec,
            coop_pops=None,
        )

    def tell(self, state: State, fitness: jax.Array) -> State:
        fitness = fitness.reshape((self.num_subpops, -1))
        pop_size = fitness.shape[1]

        state = use_state(vmap(self.base_algorithms.__class__.tell))(
            self.base_algorithms, state, fitness
        )
        min_fit_each_subpop = jnp.min(fitness, axis=1)
        best_each_subpop = jnp.argmin(fitness, axis=1)

        best_dec_this_gen = state.coop_pops.reshape(
            (self.num_subpops, pop_size, self.dim)
        )[jnp.arange(self.num_subpops), best_each_subpop, :]

        if self.random_subpop:
            # if random_subpop is set, permutate the decision variables.
            best_dec_this_gen = best_dec_this_gen[:, state.permutation]

        best_dec_this_gen = best_dec_this_gen.reshape(
            (self.num_subpops, self.num_subpops, -1)
        )[jnp.arange(self.num_subpops), jnp.arange(self.num_subpops), :]

        best_dec = jnp.where(
            (state.best_fit > min_fit_each_subpop)[:, jnp.newaxis],
            best_dec_this_gen,
            state.best_dec.reshape(self.num_subpops, -1),
        ).reshape(self.dim)

        return state.update(
            best_dec=best_dec,
            best_fit=jnp.minimum(state.best_fit, min_fit_each_subpop),
            coop_pops=None,
        )


@jit_class
@dataclass
class Coevolution(Algorithm):
    """A container for co-evolutionary algorithms.

    Parameters
    ----------
    base_algorithms:
        A batch of base algorithms.
        Usually created from `Stateful.stack(...)`, where ... is a list of algorithms.
    dim:
        The dimension of the problem in full.
    num_subpops:
        The number of subpopulations.
    random_subpop:
        Whether to shuffle the problem dimension before co-evolution.
        When set to False, each subpopulation will corespond to a contiguous block of the decision variables,
        for example, dimension 0~9 for subpopulation 0, 10~19 for subpopulation 1, etc.
        When set to True, the decision variables will be shuffled.
    """
    base_algorithms: Stack[Algorithm]
    dim: Static[int]
    num_subpops: Static[int]
    random_subpop: Static[bool]
    dtype: Static[jnp.dtype] = jnp.float32

    def setup(self, key: jax.Array) -> State:
        if self.random_subpop:
            key, subkey = jax.random.split(key)
            permutation = jax.random.permutation(subkey, self.dim)
        else:
            permutation = None

        best_dec = jnp.empty((self.dim,), dtype=self.dtype)
        best_fit = jnp.full((self.num_subpops,), jnp.inf)  # fitness

        return State(
            coop_pops=None,
            best_dec=best_dec,
            best_fit=best_fit,
            iter_counter=0,
            permutation=permutation,
        )

    def init_ask(self, state: State):
        init_subpops, state = use_state(vmap(self.base_algorithms.__class__.init_ask))(
            self.base_algorithms, state
        )
        # init_subpops (num_subpops, subpop_size, sub_dim)
        init_pop = init_subpops.transpose((1, 0, 2)).reshape(-1, self.dim)

        if self.random_subpop:
            init_pop = init_pop.at[:, state.permutation].set(init_pop)

        return init_pop, state.update(coop_pops=init_pop)

    def ask(self, state: State) -> Tuple[jax.Array, State]:
        subpop_index = state.iter_counter % self.num_subpops

        subpop, state = use_state(self.base_algorithms.__class__.ask, subpop_index)(
            self.base_algorithms, state
        )

        subpop_size = subpop.shape[0]
        # co-operate
        tiled_best_dec = jnp.tile(state.best_dec, (subpop_size, 1))

        subproblem_dim = self.dim // self.num_subpops
        coop_pops = jax.lax.dynamic_update_slice(
            tiled_best_dec, subpop, (0, subpop_index * subproblem_dim)
        )

        # if random_subpop is set, do a inverse permutation here.
        if self.random_subpop:
            coop_pops = coop_pops.at[:, state.permutation].set(coop_pops)

        return coop_pops, state.update(coop_pops=coop_pops)

    def init_tell(self, state, fitness):
        best_fit = jnp.min(fitness)
        state = use_state(
            vmap(self.base_algorithms.__class__.init_tell, in_axes=(0, 0, None))
        )(self.base_algorithms, state, fitness)
        best_dec = state.coop_pops[jnp.argmin(fitness)]
        if self.random_subpop:
            best_dec = best_dec[state.permutation]

        return state.update(
            best_fit=jnp.tile(best_fit, self.num_subpops),
            best_dec=best_dec,
            coop_pops=None,
        )

    def tell(self, state: State, fitness: jax.Array) -> State:
        subpop_index = state.iter_counter % self.num_subpops
        state = use_state(self.base_algorithms.__class__.tell, subpop_index)(
            self.base_algorithms, state, fitness
        )
        min_fitness = jnp.min(fitness)

        best_dec_this_gen = state.coop_pops[jnp.argmin(fitness)]
        if self.random_subpop:
            # if random_subpop is set, permutate the decision variables.
            best_dec_this_gen = best_dec_this_gen[state.permutation]

        best_dec = jax.lax.select(
            state.best_fit[subpop_index] > min_fitness,
            best_dec_this_gen,
            state.best_dec,
        )

        best_fit = state.best_fit.at[subpop_index].min(min_fitness)

        return state.update(
            best_dec=best_dec,
            best_fit=best_fit,
            iter_counter=state.iter_counter + 1,
            coop_pops=None,
        )

from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from evox import Algorithm, State, jit_class
from jax import vmap
from jax.tree_util import tree_map


@jit_class
class VectorizedCoevolution(Algorithm):
    """
    Automatically apply cooperative coevolution to any algorithm.
    The process of cooperative coevolution is vectorized,
    meaning all sub-populations will evolve at the same time in each generation.
    """

    def __init__(
        self,
        base_algorithm: Algorithm,
        dim: int,
        num_subpops: int,
        random_subpop: bool = True,
    ):
        assert dim % num_subpops == 0

        self._base_algorithm = base_algorithm
        self.dim = dim
        self.num_subpops = num_subpops
        self.random_subpop = random_subpop
        self.sub_dim = dim // num_subpops

    def setup(self, key: jax.Array) -> State:
        if self.random_subpop:
            key, subkey = jax.random.split(key)
            self.permutation = jax.random.permutation(subkey, self.dim)

        best_dec = jnp.empty((self.dim,))
        best_fit = jnp.inf  # fitness
        keys = jax.random.split(key, self.num_subpops)
        base_alg_state = vmap(self._base_algorithm.init)(keys)

        return State(
            first_iter=True,
            best_dec=best_dec,
            best_fit=best_fit,
            base_alg_state=base_alg_state,
        )

    def ask(self, state: State) -> Tuple[jax.Array, State]:
        subpops, base_alg_state = vmap(self._base_algorithm.ask)(state.base_alg_state)

        # in the first iteration, we don't really have a best solution
        # so just pick the first solution.
        best_dec = jax.lax.select(
            state.first_iter, subpops[:, 0, :].reshape((self.dim,)), state.best_dec
        )
        _num_subpops, pop_size, _sub_dim = subpops.shape
        # co-operate
        tiled_best_dec = jnp.tile(best_dec, (pop_size, 1))
        coop_pops = vmap(
            lambda index: jax.lax.dynamic_update_slice(
                tiled_best_dec, subpops[index], (0, index * self.sub_dim)
            )
        )(jnp.arange(self.num_subpops))
        coop_pops = coop_pops.reshape((self.num_subpops * pop_size, self.dim))

        # if random_subpop is set, do a inverse permutation here.
        if self.random_subpop:
            coop_pops = coop_pops.at[:, self.permutation].set(coop_pops)

        return coop_pops, state.update(
            first_iter=False, base_alg_state=base_alg_state, coop_pops=coop_pops
        )

    def tell(self, state: State, fitness: jax.Array) -> State:
        fitness = fitness.reshape((self.num_subpops, -1))
        base_alg_state = vmap(self._base_algorithm.tell)(state.base_alg_state, fitness)
        min_fitness = jnp.min(fitness)

        best_dec_this_gen = state.coop_pops[jnp.argmin(fitness)]
        if self.random_subpop:
            # if random_subpop is set, permutate the decision variables.
            best_dec_this_gen = best_dec_this_gen[self.permutation]

        best_dec = jax.lax.select(
            state.best_fit > min_fitness,
            best_dec_this_gen,
            state.best_dec,
        )

        return state.update(
            base_alg_state=base_alg_state,
            best_dec=best_dec,
            best_fit=jnp.minimum(state.best_fit, min_fitness),
        )


@jit_class
class Coevolution(Algorithm):
    """
    Automatically apply cooperative coevolution to any algorithm.
    The process of cooperative coevolution is not vectorized,
    meaning all sub-populations will evolve one at a time (round-robin) in each generation.
    """

    def __init__(
        self,
        base_algorithm: Algorithm,
        dim: int,
        num_subpops: int,
        subpop_size: int,
        num_subpop_iter: int = 1,
        random_subpop: bool = True,
    ):
        assert dim % num_subpops == 0

        self._base_algorithm = base_algorithm
        self.dim = dim
        self.num_subpops = num_subpops
        self.subpop_size = subpop_size
        self.num_subpop_iter = num_subpop_iter
        self.random_subpop = random_subpop
        self.sub_dim = dim // num_subpops

    def setup(self, key: jax.Array) -> State:
        if self.random_subpop:
            key, subkey = jax.random.split(key)
            self.permutation = jax.random.permutation(subkey, self.dim)

        best_dec = jnp.empty((self.dim,))
        best_fit = jnp.full((self.num_subpops,), jnp.inf)  # fitness
        keys = jax.random.split(key, self.num_subpops)
        base_alg_state = vmap(self._base_algorithm.init)(keys)

        return State(
            iter_counter=0,
            subpops=jnp.empty((self.num_subpops, self.subpop_size, self.sub_dim)),
            best_dec=best_dec,
            best_fit=best_fit,
            base_alg_state=base_alg_state,
        )

    def ask(self, state: State) -> Tuple[jax.Array, State]:
        subpop_index = (state.iter_counter // self.num_subpop_iter) % self.num_subpops

        # Ask all the algorithms once to initialize the best solution,
        def init_best_dec(state):
            # in the first iteration, we don't really have a best solution
            # so just pick the first solution.
            init_subpops, _base_alg_state = vmap(self._base_algorithm.ask)(state.base_alg_state)
            first_dec = init_subpops[:, 0, :]
            return first_dec.reshape((self.dim,))

        best_dec = jax.lax.cond(
            state.iter_counter == 0,
            init_best_dec,
            lambda state: state.best_dec,
            state,
        )

        subpop, sub_alg_state = self._base_algorithm.ask(
            state.base_alg_state[subpop_index]
        )
        subpops = state.subpops.at[subpop_index].set(subpop)
        base_alg_state = tree_map(
            lambda old, new: old.at[subpop_index].set(new),
            state.base_alg_state,
            sub_alg_state,
        )

        # co-operate
        tiled_best_dec = jnp.tile(best_dec, (self.subpop_size, 1))
        coop_pops = jax.lax.dynamic_update_slice(
            tiled_best_dec, subpop, (0, subpop_index * self.sub_dim)
        )

        # if random_subpop is set, do a inverse permutation here.
        if self.random_subpop:
            coop_pops = coop_pops.at[:, self.permutation].set(coop_pops)

        return coop_pops, state.update(
            subpops=subpops,
            base_alg_state=base_alg_state,
            coop_pops=coop_pops,
        )

    def tell(self, state: State, fitness: jax.Array) -> State:
        subpop_index = (state.iter_counter // self.num_subpop_iter) % self.num_subpops
        subpop_base_alg_state = self._base_algorithm.tell(
            state.base_alg_state[subpop_index], fitness
        )
        base_alg_state = tree_map(
            lambda old, new: old.at[subpop_index].set(new),
            state.base_alg_state,
            subpop_base_alg_state,
        )
        min_fitness = jnp.min(fitness)

        best_dec_this_gen = state.coop_pops[jnp.argmin(fitness)]
        if self.random_subpop:
            # if random_subpop is set, permutate the decision variables.
            best_dec_this_gen = best_dec_this_gen[self.permutation]

        best_dec = jax.lax.select(
            state.best_fit[subpop_index] > min_fitness,
            best_dec_this_gen,
            state.best_dec,
        )

        best_fit = state.best_fit.at[subpop_index].min(min_fitness)

        return state.update(
            base_alg_state=base_alg_state,
            best_dec=best_dec,
            best_fit=best_fit,
            iter_counter=state.iter_counter + 1,
        )

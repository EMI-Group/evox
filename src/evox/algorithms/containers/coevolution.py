from dataclasses import field
from functools import partial
from typing import List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import vmap
from jax.tree_util import tree_map

from evox import Algorithm, State, Static, Stack, dataclass, jit_class, use_state


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


def coevolution(
    base_algorithms,
    dim,
    num_subpops,
    subpop_size,
    random_subpop=True,
    dtype=jnp.float32,
):
    subproblem_dim = dim // num_subpops
    algorithm_class = base_algorithms.__class__

    @jit_class
    @dataclass
    class Coevolution(Algorithm):
        base_algorithms: Stack[Algorithm]
        dim: Static[int]
        num_subpops: Static[int]
        subpop_size: Static[int]
        random_subpop: Static[bool]

        def setup(self, key: jax.Array) -> State:
            if self.random_subpop:
                key, subkey = jax.random.split(key)
                permutation = jax.random.permutation(subkey, self.dim)
            else:
                permutation = None

            best_dec = jnp.empty((self.dim,), dtype=dtype)
            best_fit = jnp.full((self.num_subpops,), jnp.inf)  # fitness

            return State(
                coop_pops=jnp.empty((self.subpop_size, self.dim)),
                best_dec=best_dec,
                best_fit=best_fit,
                iter_counter=0,
                permutation=permutation,
            )

        def init_ask(self, state: State):
            init_subpops, state = use_state(vmap(algorithm_class.init_ask))(
                base_algorithms, state
            )
            # init_subpops (num_subpops, subpop_size, sub_dim)
            init_pop = init_subpops.transpose((1, 0, 2)).reshape(-1, self.dim)
            return init_pop, state.update(coop_pops=init_pop)

        def ask(self, state: State) -> Tuple[jax.Array, State]:
            subpop_index = state.iter_counter % self.num_subpops

            subpop, state = use_state(algorithm_class.ask, subpop_index)(
                base_algorithms, state
            )

            # co-operate
            tiled_best_dec = jnp.tile(state.best_dec, (self.subpop_size, 1))
            coop_pops = jax.lax.dynamic_update_slice(
                tiled_best_dec, subpop, (0, subpop_index * subproblem_dim)
            )

            # if random_subpop is set, do a inverse permutation here.
            if self.random_subpop:
                coop_pops = coop_pops.at[:, state.permutation].set(coop_pops)

            return coop_pops, state.update(coop_pops=coop_pops)

        def init_tell(self, state, fitness):
            best_fit = jnp.min(fitness)
            state = use_state(vmap(algorithm_class.init_tell, in_axes=(0, 0, None)))(
                base_algorithms, state, fitness
            )
            best_dec = state.coop_pops[jnp.argmin(fitness)]
            return state.update(
                best_fit=jnp.tile(best_fit, self.num_subpops),
                coop_pops=None,
            )

        def tell(self, state: State, fitness: jax.Array) -> State:
            subpop_index = state.iter_counter % self.num_subpops
            state = use_state(algorithm_class.tell, subpop_index)(
                base_algorithms, state, fitness
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

    return Coevolution(base_algorithms, dim, num_subpops, subpop_size, random_subpop)


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

from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from evox import Algorithm, State
from jax import vmap


class VectorizedCoevolution(Algorithm):
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
        base_alg_state = vmap(
            partial(self._base_algorithm.init, name="base_algorithm")
        )(keys)

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

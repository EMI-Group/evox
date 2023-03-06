from functools import partial

import jax
import jax.numpy as jnp
from evox import Algorithm, State
from jax import vmap
from jax.tree_util import tree_map


class ClusterdAlgorithm(Algorithm):
    """A container that split the encoding into subproblems, and run an Algorithm on each.

    Can take in any base algorithm, split the problem into n different sub-problems
    and solve each problem using the base algorithm.
    Dim must be a multiple of num_cluster, and num_cluster must be a multiple of num_gpus.
    """

    def __init__(
        self,
        base_algorithm: Algorithm,
        dim: int,
        num_cluster: int,
    ):
        assert dim % num_cluster == 0
        self.dim = dim
        self.num_cluster = num_cluster
        self.subproblem_dim = self.dim // num_cluster
        self._base_algorithm = base_algorithm

    def setup(self, key: jax.Array) -> State:
        keys = jax.random.split(key, self.num_cluster)
        vectorized_state = vmap(self._base_algorithm.init)(keys)
        return vectorized_state

    def ask(self, state: State):
        sub_pops, state = vmap(self._base_algorithm.ask)(state)
        # concatenate different parts as a whole
        full_pop = jnp.concatenate(sub_pops, axis=1)
        return full_pop, state

    def tell(self, state: State, fitness: jax.Array):
        return vmap(self._base_algorithm.tell, in_axes=(0, None))(state, fitness)


def _mask_state(state: State, permutation: jax.Array):
    return tree_map(
        lambda x: x[permutation, ...],
        state,
    )


def _unmask_state(old_state: State, new_state: State, permutation: jax.Array):
    assert isinstance(old_state, State)
    assert isinstance(new_state, State)
    return tree_map(
        lambda old_value, new_value: old_value.at[permutation, ...].set(new_value),
        old_state,
        new_state,
    )


class RandomMaskAlgorithm(Algorithm):
    """Cluster container with random mask

    pop_size is needed, because JAX needs static shape,
    if pop_size is None, it will try to read pop_size from base_algorithm

    """

    def __init__(
        self,
        base_algorithm: Algorithm,
        dim: int,
        num_cluster: int,
        num_mask: int = 1,
        change_every: int = 1,
        pop_size=None,
    ):
        assert dim % num_cluster == 0
        assert 0 < num_mask < num_cluster
        self.dim = dim
        self.num_cluster = num_cluster
        self.num_mask = num_mask
        self.num_valid = num_cluster - num_mask
        self.change_every = change_every
        self.subproblem_dim = self.dim // num_cluster
        self.submodule_name = "_base_algorithm"
        self.base_algorithm = base_algorithm
        if pop_size is None:
            self.pop_size = base_algorithm.pop_size
        else:
            self.pop_size = pop_size

    def init(self, key: jax.Array = None, name: str = "_top_level"):
        self.name = name
        keys = jax.random.split(key, self.num_cluster)
        child_states = {self.submodule_name: vmap(self.base_algorithm.init)(keys)}
        self_state = self.setup(key)
        return self_state._set_child_states(child_states)

    def setup(self, key: jax.Array):
        return State(
            key=key,
            sub_pops=jnp.zeros((self.num_cluster, self.pop_size, self.subproblem_dim)),
            permutation=jnp.arange(self.num_valid),
            count=0,
        )

    def ask(self, state: State):
        state = self._try_change_mask(state)
        return jax.lax.cond(state.count == 0, self._ask_init, self._ask_normal, state)

    def _ask_init(self, state: State):
        child_state, xs = vmap(self.base_algorithm.ask)(
            state.get_child_state(self.submodule_name)
        )
        state = state.update_child(self.submodule_name, child_state)
        # concatenate different parts as a whole
        pop = jnp.concatenate(xs, axis=1)
        return pop, state.update(sub_pops=xs)

    def _ask_normal(self, state: State):
        old_state = state.get_child_state(self.submodule_name)
        masked_child_state = _mask_state(old_state, state.permutation)
        new_child_state, xs = vmap(self.base_algorithm.ask)(masked_child_state)
        full_pop = jnp.concatenate(state.sub_pops.at[state.permutation].set(xs), axis=1)
        state = state.update_child(
            self.submodule_name,
            _unmask_state(old_state, new_child_state, state.permutation),
        )
        return full_pop, state

    def tell(self, state: State, fitness: jax.Array):
        old_state = state.get_child_state(self.submodule_name)
        masked_child_state = _mask_state(old_state, state.permutation)
        new_child_state = vmap(self.base_algorithm.tell, in_axes=(0, None))(
            masked_child_state, fitness
        )
        state = state.update_child(
            self.submodule_name,
            _unmask_state(old_state, new_child_state, state.permutation),
        )
        return state.update(count=state.count + 1)

    def _try_change_mask(self, state: State):
        def change_mask(state: State):
            key, subkey = jax.random.split(state.key)
            return state.update(
                key=key,
                permutation=jax.random.choice(
                    subkey, self.num_cluster, (self.num_valid,)
                ),
                count=0,
            )

        return jax.lax.cond(
            state.count % self.change_every == 0,
            lambda state: state,
            change_mask,
            state,
        )

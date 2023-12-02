from evox import jit_class, Operator
from evox.core.state import State
import jax
import jax.numpy as jnp
from jax import jit

from functools import partial


@partial(jit, static_argnums=[1])
def select_rand_pbest(key, percent, population, fitness):
    assert percent > 0 and percent <= 1.0
    pop_size = population.shape[0]
    top_p_num = max(int(pop_size * percent), 1)  # The number is p,  ranges in 5% - 20%.
    sorted_indices = jnp.argsort(fitness)
    _topk_fit, pbest_indices = jax.lax.top_k(-fitness, top_p_num)
    pbest_index = jax.random.choice(key, pbest_indices)
    pbest_vect = population[pbest_index]
    return pbest_vect

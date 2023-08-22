from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, lax, random

from evox import jit_class


def _de_mutation(x1, x2, x3, F):
    mutated_pop = x1 + F * (x2 - x3)
    return mutated_pop


def _de_crossover(key, new_x, x, CR):
    batch, dim = x.shape
    random_crossover = random.uniform(key, shape=(batch, dim))
    mask = random_crossover < CR
    return jnp.where(mask, new_x, x)


@jit
def differential_evolve(key, x1, x2, x3, F, CR):
    key, de_key = random.split(key)
    mutated_pop = _de_mutation(x1, x2, x3, F)

    children = _de_crossover(de_key, mutated_pop, x1, CR)
    return children


@jit_class
class DifferentialEvolve:
    def __init__(self, F=0.5, CR=1):
        """
        Parameters
        ----------
        F
            The scaling factor
        CR
            The probability of crossover
        """
        self.F = F
        self.CR = CR

    def __call__(self, key, p1, p2, p3):
        return differential_evolve(key, p1, p2, p3, self.F, self.CR)


@partial(jit, static_argnums=[1])
def de_diff_sum(key, diff_padding_num, num_diff_vects, index, population):
    """Make differences and sum"""
    # Randomly select 1 random individual (first index) and (num_diff_vects * 2) difference individuals
    pop_size, dim = population.shape
    select_len = num_diff_vects * 2 + 1
    random_choice = jax.random.choice(
        key, pop_size, shape=(diff_padding_num,), replace=False
    )
    random_choice = jnp.where(
        random_choice == index, pop_size - 1, random_choice
    )  # Ensure that selected indices != index

    # Permutate indices, take the first select_len individuals of indices in the population, and set the next individuals to 0
    pop_permut = population[random_choice]
    permut_mask = jnp.where(jnp.arange(diff_padding_num) < select_len, True, False)
    pop_permut_padding = jnp.where(
        permut_mask[:, jnp.newaxis], pop_permut, jnp.zeros((diff_padding_num, dim))
    )

    diff_vects = pop_permut_padding[1:, :]
    subtrahend_index = jnp.arange(1, diff_vects.shape[0], 2)
    difference_sum = jnp.sum(diff_vects.at[subtrahend_index, :].multiply(-1), axis=0)

    rand_vect_idx = random_choice[0]
    return (difference_sum, rand_vect_idx)


@jit
def de_bin_cross(key, mutation_vector, current_vect, CR):
    # Binary crossover: dimension-by-dimension crossover
    # , based on cross_probability to determine the crossover needed for that dimension.
    R_key, mask_key = jax.random.split(key, 2)
    dim = mutation_vector.shape[0]
    R = jax.random.choice(
        R_key, dim
    )  # R is the jrand, i.e. the dimension that must be changed in the crossover
    mask = jax.random.uniform(mask_key, shape=(dim,)) < CR
    mask = mask.at[R].set(True)

    trial_vector = jnp.where(
        mask,
        mutation_vector,
        current_vect,
    )
    return trial_vector


@jit
def de_exp_cross(key, mutation_vector, current_vect, CR):
    # Exponential crossover: Cross the n-th to (n+l-1)-th dimension of the vector,
    # and if n+l-1 exceeds the maximum dimension dim, then make it up from the beginning

    n_key, l_key = jax.random.split(key, 2)
    dim = mutation_vector.shape[0]
    n = jax.random.choice(n_key, jnp.arange(dim))

    # Generate l according to CR. n is the starting dimension to be crossover, and l is the crossover length
    l_mask = jax.random.uniform(l_key, shape=(dim,)) < CR

    def count_forward_true(i, l, l_mask):
        # count_forward_true() is used to count the number of preceding trues (randnum < cr)
        replace_vect_init = jnp.arange(dim)
        replace_vect = jnp.where(replace_vect_init > i, True, False)
        forward_mask = jnp.logical_or(replace_vect, l_mask)
        forward_bool = jnp.all(forward_mask)
        l = lax.select(forward_bool, i + 1, l)
        return l

    l = lax.fori_loop(0, dim, partial(count_forward_true, l_mask=l_mask), init_val=0)
    # Generate mask by n and l
    mask_init = jnp.arange(dim)
    mask_bin = jnp.where(mask_init < l, True, False)
    mask = jnp.roll(mask_bin, shift=n)
    trial_vector = jnp.where(
        mask,
        mutation_vector,
        current_vect,
    )
    return trial_vector


@jit
def de_arith_recom(mutation_vector, current_vect, K):
    # K can take CR
    trial_vector = current_vect + K * (mutation_vector - current_vect)
    return trial_vector

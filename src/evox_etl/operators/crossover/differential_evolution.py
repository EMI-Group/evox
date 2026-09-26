from typing import Tuple

import etl
import etl.numpy as enp
import etl.random as random


def DE_differential_sum(
    key,
    diff_padding_num: int,
    num_diff_vectors,
    index,
    population,
    F=None,
    replace: bool = False,
) -> Tuple:
    """
    Computes the difference vectors' sum in differential evolution.

    :param key: The random key.
    :param diff_padding_num: The number of padding difference vectors.
    :param num_diff_vectors: The number of difference vectors used in mutation.
    :param index: The index of current individual.
    :param population: The population tensor.
    :param F: The mutation scale factor (scalar or (pop,) tensor). None -> old behavior.
    :param replace: Whether an individual may be picked multiple times.

    :return: The difference sum and the index of first difference vector.
    """
    pop_size = population.shape[0]
    if getattr(num_diff_vectors, "shape", None) is not None and len(num_diff_vectors.shape) == 0:
        num_diff_vectors = etl.reshape(num_diff_vectors, (1,))

    select_len = enp.expand_dims(etl.cast(num_diff_vectors, "int32"), 1) * 2 + 1
    rand_indices = random.randint(key, (pop_size, diff_padding_num), 0, pop_size, "int32")
    if not replace:
        # Ensure the current individual is never picked (mirror torch fix).
        rand_indices = etl.select(
            rand_indices == enp.expand_dims(etl.cast(index, "int32"), 1), pop_size - 1, rand_indices
        )

    pop_permute = etl.gather(population, rand_indices, axis=0)
    mask = enp.expand_dims(enp.arange(diff_padding_num, dtype="int32"), 0) < select_len
    pop_permute_padding = etl.select(enp.expand_dims(mask, 2), pop_permute, 0.0)

    diff_vectors = pop_permute_padding[:, 1:]
    difference_sum = enp.sum(diff_vectors[:, 0::2], axis=1) - enp.sum(diff_vectors[:, 1::2], axis=1)
    if F is not None:
        if getattr(F, "shape", None) is not None and len(F.shape) == 1:
            F = enp.expand_dims(F, 1)
        difference_sum = difference_sum * F
    return difference_sum, rand_indices[:, 0]


def DE_binary_crossover(key, mutation_vector, current_vector, CR):
    """
    Performs binary crossover in differential evolution.

    :param key: The random key.
    :param mutation_vector: The mutated vector for each individual in the population.
    :param current_vector: The current vector for each individual in the population.
    :param CR: The crossover probability for each individual.

    :return: The trial vector after crossover for each individual.
    """
    k1, k2 = random.split(key)
    pop_size, dim = mutation_vector.shape
    if getattr(CR, "shape", None) is not None and len(CR.shape) == 1:
        CR = enp.expand_dims(CR, 1)
    rn = random.normal(k1, (pop_size, dim), 0.0, 1.0, "float32")
    mask = rn < CR
    rind = enp.expand_dims(random.randint(k2, (pop_size,), 0, dim, "int32"), 1)
    jind = enp.expand_dims(enp.arange(dim, dtype="int32"), 0) == rind
    trial_vector = etl.select(etl.logical_or(mask, jind), mutation_vector, current_vector)
    return trial_vector


def DE_exponential_crossover(key, mutation_vector, current_vector, CR):
    """
    Performs exponential crossover in differential evolution.

    :param key: The random key.
    :param mutation_vector: The mutated vector for each individual in the population.
    :param current_vector: The current vector for each individual in the population.
    :param CR: The crossover probability for each individual.

    :return: The trial vector after crossover for each individual.
    """
    pop_size, dim = mutation_vector.shape
    k1, k2 = random.split(key)
    nn = random.randint(k1, (pop_size,), 0, dim, "int32")
    # Geometric distribution random ll
    float_tiny = 1.1754943508222875e-38
    ll = random.uniform(k2, (pop_size,), dtype="float32")
    ll = enp.maximum(ll, float_tiny)
    ll = etl.cast(etl.floor(enp.log(ll) / (-etl.log1p(CR))), "int32")
    ll = enp.minimum(ll, dim)
    mask = enp.expand_dims(enp.arange(dim, dtype="int32"), 0) < enp.expand_dims(ll - 1, 1)
    tiled = etl.tile(mask, (1, 2))
    idx = enp.expand_dims(etl.cast(nn, "int64"), 1) + enp.expand_dims(enp.arange(dim, dtype="int64"), 0)
    idx_flat = enp.expand_dims(enp.arange(pop_size, dtype="int64") * (2 * dim), 1) + idx
    mask = etl.gather(etl.reshape(tiled, (-1,)), idx_flat)
    trial_vector = etl.select(mask, mutation_vector, current_vector)
    return trial_vector


def DE_arithmetic_recombination(mutation_vector, current_vector, K):
    """
    Performs arithmetic recombination in differential evolution.

    :param mutation_vector: The mutated vector for each individual in the population.
    :param current_vector: The current vector for each individual in the population.
    :param K: The coefficient for each individual.

    :return: The trial vector after recombination for each individual.
    """
    if getattr(K, "shape", None) is not None and len(K.shape) == 1:
        K = enp.expand_dims(K, 1)
    trial_vector = current_vector + K * (mutation_vector - current_vector)
    return trial_vector

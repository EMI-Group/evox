"""Crossover operator shims: torch `evox.operators.crossover` ported to etl.

Temporary single-file port under `evox_etl.algorithms` so in-flight algorithm
ports can import crossover ops before `evox_etl.operators.crossover` lands.
Plain functions (no `@etl.defn`, per DESIGN.md §4.3) — call them only inside
an active trace. Functions that draw randomness in torch take `key` FIRST.
"""
from typing import Optional, Union

import etl
import etl.numpy as enp
import etl.random as random

from etl import core


def simulated_binary(key, x, pro_c: float = 1.0, dis_c: float = 20.0) -> core.SymbolicTensor:
    """Simulated binary crossover (SBX): pair x[i] with x[i + n//2] and
    return both offspring per pair, concatenated along dim 0."""
    n, m = x.shape
    half = n // 2
    parent1_dec = etl.slice(x, (0, 0), (half, m), (1, 1))
    parent2_dec = etl.slice(x, (half, 0), (half, m), (1, 1))

    # torch draws, in order: mu (rand), sign (randint 0..2), two rand gates.
    k_mu, k_sign, k_gate1, k_gate2 = random.split_n(key, 4)

    mu = random.uniform(k_mu, (half, m), 0.0, 1.0, "float32")

    # Beta calculation for SBX (torch.pow -> etl.power, top-level export of
    # etl.ops.elementwise.power).
    beta = etl.select(
        mu <= 0.5,
        etl.power(2.0 * mu, 1.0 / (dis_c + 1.0)),
        etl.power(2.0 - 2.0 * mu, -1.0 / (dis_c + 1.0)),
    )

    # Random binary for mutation direction: 1 - randint(0, 2) * 2.
    sign = 1.0 - 2.0 * etl.cast(random.randint(k_sign, (half, m), 0, 2, "int32"), "float32")
    beta = beta * sign

    # Apply crossover probability to mutate.
    beta = etl.select(random.uniform(k_gate1, (half, m), 0.0, 1.0, "float32") < 0.5, 1.0, beta)
    beta = etl.select(random.uniform(k_gate2, (half, m), 0.0, 1.0, "float32") > pro_c, 1.0, beta)

    mid = (parent1_dec + parent2_dec) / 2.0
    half_diff = (parent1_dec - parent2_dec) / 2.0
    return etl.concatenate([mid + beta * half_diff, mid - beta * half_diff], axis=0)


def simulated_binary_half(key, x, pro_c: float = 1.0, dis_c: float = 20.0) -> core.SymbolicTensor:
    """SBX returning only the first offspring of each parent pair."""
    n, m = x.shape
    half = n // 2
    parent1_dec = etl.slice(x, (0, 0), (half, m), (1, 1))
    parent2_dec = etl.slice(x, (half, 0), (half, m), (1, 1))

    k_mu, k_sign, k_gate1, k_gate2 = random.split_n(key, 4)

    mu = random.uniform(k_mu, (half, m), 0.0, 1.0, "float32")

    beta = etl.select(
        mu <= 0.5,
        etl.power(2.0 * mu, 1.0 / (dis_c + 1.0)),
        etl.power(2.0 - 2.0 * mu, -1.0 / (dis_c + 1.0)),
    )

    sign = 1.0 - 2.0 * etl.cast(random.randint(k_sign, (half, m), 0, 2, "int32"), "float32")
    beta = beta * sign

    beta = etl.select(random.uniform(k_gate1, (half, m), 0.0, 1.0, "float32") < 0.5, 1.0, beta)
    beta = etl.select(random.uniform(k_gate2, (half, m), 0.0, 1.0, "float32") > pro_c, 1.0, beta)

    return (parent1_dec + parent2_dec) / 2.0 + beta * (parent1_dec - parent2_dec) / 2.0


def _minimum_int(a, b: int):
    """Elementwise minimum of int tensor `a` and int `b` — 1:1 port of
    `evox.utils.minimum_int` (exact for integer dtypes; the torch version's
    inaccuracy only affects float inputs, which never occur here)."""
    return etl.minimum(a, b)


def DE_differential_sum(
    key,
    diff_padding_num: int,
    num_diff_vectors,
    index,
    population,
    F: Optional[Union[float, core.SymbolicTensor]] = None,
    replace: bool = False,
):
    """Sum of the difference vectors for DE mutation; returns (difference_sum
    (pop, dim), first sampled index (pop,)). replace=False (default) remaps
    self-picks to the last index exactly like torch; non-None F scales the sum."""
    pop_size, dim = population.shape
    if len(getattr(num_diff_vectors, "shape", ())) == 0:
        num_diff_vectors = etl.reshape(num_diff_vectors, (1,))

    select_len = enp.expand_dims(num_diff_vectors, 1) * 2 + 1
    rand_indices = random.randint(key, (pop_size, diff_padding_num), 0, pop_size, "int32")
    if not replace:
        # torch.where(rand_indices == index[:, None], pop_size - 1, rand_indices).
        # An int32 constant branch keeps the result int32 (etl promotes a bare
        # Python int against an int tensor to int64).
        cond = rand_indices == enp.expand_dims(index, 1)
        rand_indices = etl.select(
            cond, enp.full(rand_indices.shape, pop_size - 1, dtype="int32"), rand_indices
        )

    pop_permute = etl.gather(population, rand_indices, axis=0)
    mask = enp.expand_dims(enp.arange(diff_padding_num, dtype="int32"), 0) < select_len
    pop_permute_padding = etl.select(enp.expand_dims(mask, 2), pop_permute, 0.0)

    diff_vectors = etl.slice(
        pop_permute_padding, (0, 1, 0), (pop_size, diff_padding_num - 1, dim), (1, 1, 1)
    )
    even = etl.slice(
        diff_vectors, (0, 0, 0), (pop_size, diff_padding_num // 2, dim), (1, 2, 1)
    )
    odd = etl.slice(
        diff_vectors, (0, 1, 0), (pop_size, (diff_padding_num - 1) // 2, dim), (1, 2, 1)
    )
    difference_sum = enp.sum(even, axis=1) - enp.sum(odd, axis=1)

    if F is not None:
        if len(getattr(F, "shape", ())) == 1:
            F = enp.expand_dims(F, 1)
        difference_sum = difference_sum * F

    first = etl.slice(rand_indices, (0, 0), (pop_size, 1), (1, 1))
    return difference_sum, etl.reshape(first, (pop_size,))


def DE_binary_crossover(key, mutation_vector, current_vector, CR) -> core.SymbolicTensor:
    """Binary crossover: take each dim from the mutant with prob CR, plus
    one forced mutant dim per row (rind)."""
    k_mask, k_rind = random.split(key)
    pop_size, dim = mutation_vector.shape
    if len(CR.shape) == 1:
        CR = enp.expand_dims(CR, 1)
    mask = random.normal(k_mask, (pop_size, dim), 0.0, 1.0, "float32") < CR
    rind = enp.expand_dims(random.randint(k_rind, (pop_size,), 0, dim, "int32"), 1)
    jind = enp.expand_dims(enp.arange(dim, dtype="int32"), 0) == rind
    return etl.select(etl.logical_or(mask, jind), mutation_vector, current_vector)


def DE_exponential_crossover(key, mutation_vector, current_vector, CR) -> core.SymbolicTensor:
    """Exponential crossover: swap a circular run of dims starting at a
    random index; run length ~ geometric(CR) (torch-exact off-by-ones)."""
    pop_size, dim = mutation_vector.shape
    k_nn, k_ll = random.split(key)
    nn = random.randint(k_nn, (pop_size,), 0, dim, "int32")

    # Geometric distribution random ll (torch.clamp(min=float_tiny) ->
    # enp.maximum; minimum_int -> _minimum_int).
    float_tiny = 1.1754943508222875e-38
    ll = random.uniform(k_ll, (pop_size,), 0.0, 1.0, "float32")
    ll = enp.maximum(ll, float_tiny)
    ll = etl.cast(etl.floor(enp.log(ll) / (-etl.log1p(CR))), "int32")
    ll = _minimum_int(ll, dim)

    base_mask = enp.expand_dims(enp.arange(dim, dtype="int32"), 0) < enp.expand_dims(ll - 1, 1)
    # torch.gather(tile(mask, (1, 2)), 1, nn[:, None] + arange(dim)): etl.gather
    # is numpy-take style, so flatten the tiled mask and gather with flat
    # indices row_start * (2 * dim) + nn + j.
    tiled = etl.tile(base_mask, (1, 2))
    idx = enp.expand_dims(etl.cast(nn, "int64"), 1) + enp.expand_dims(enp.arange(dim, dtype="int64"), 0)
    idx_flat = enp.expand_dims(enp.arange(pop_size, dtype="int64") * (2 * dim), 1) + idx
    mask = etl.gather(etl.reshape(tiled, (-1,)), idx_flat)

    return etl.select(mask, mutation_vector, current_vector)


def DE_arithmetic_recombination(mutation_vector, current_vector, K) -> core.SymbolicTensor:
    """Arithmetic recombination: current + K * (mutant - current); K is a
    scalar or a (pop,)/(pop, 1) coefficient tensor."""
    if len(K.shape) == 1:
        K = enp.expand_dims(K, 1)
    return current_vector + K * (mutation_vector - current_vector)

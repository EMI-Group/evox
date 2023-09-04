import jax
import jax.numpy as jnp
from itertools import combinations as n_choose_k
from math import comb

import evox


@evox.jit_class
class UniformSampling:
    """
    Uniform sampling use Das and Dennis's method, Deb and Jain's method.
    Inspired by PlatEMO's NBI algorithm.
    """

    def __init__(self, n=None, m=None):
        self.n = n
        self.m = m

    def __call__(self):
        h1 = 1
        while comb(h1 + self.m, self.m - 1) <= self.n:
            h1 += 1

        w = (
            jnp.array(list(n_choose_k(range(1, h1 + self.m), self.m - 1)))
            - jnp.tile(
                jnp.array(range(self.m - 1)),
                (comb(h1 + self.m - 1, self.m - 1), 1),
            )
            - 1
        )
        w = (
            jnp.c_[w, jnp.zeros((jnp.shape(w)[0], 1)) + h1]
            - jnp.c_[jnp.zeros((jnp.shape(w)[0], 1)), w]
        ) / h1
        if h1 < self.m:
            h2 = 0
            while (
                comb(h1 + self.m - 1, self.m - 1) + comb(h2 + self.m, self.m - 1)
                <= self.n
            ):
                h2 += 1
            if h2 > 0:
                w2 = (
                    jnp.array(list(n_choose_k(range(1, h2 + self.m), self.m - 1)))
                    - jnp.tile(
                        jnp.array(range(self.m - 1)),
                        (comb(h2 + self.m - 1, self.m - 1), 1),
                    )
                    - 1
                )
                w2 = (
                    jnp.c_[w2, jnp.zeros((jnp.shape(w2)[0], 1)) + h2]
                    - jnp.c_[jnp.zeros((jnp.shape(w2)[0], 1)), w2]
                ) / h2
                w = jnp.r_[w, w2 / 2.0 + 1.0 / (2.0 * self.m)]
        w = jnp.maximum(w, 1e-6)
        n = jnp.shape(w)[0]
        return w, n

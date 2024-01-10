# --------------------------------------------------------------------------------------
# 1. This code implements algorithms described in the following papers:
#
# Title: A Cooperative approach to particle swarm optimization
# Link: https://ieeexplore.ieee.org/document/1304845
# --------------------------------------------------------------------------------------

import jax
import jax.numpy as jnp

from evox.utils import *
from evox import Algorithm, State, jit_class
from evox.algorithms.containers.coevolution import Coevolution
from evox.algorithms.so.pso_variants.pso import PSO


# CPSO-S: Cooperative PSO
@jit_class
class CPSOS(Coevolution):
    """Cooperative particle swarm optimizer.
    Implemented using EvoX's built-in coevolution framework.
    CPSOS essentially a wrapper around PSO and Coevolution.

    https://ieeexplore.ieee.org/document/1304845
    """

    def __init__(
        self,
        lb,  # lower bound of problem
        ub,  # upper bound of problem
        subpop_size: int,
        inertia_weight: float,  # w
        cognitive_coefficient: float,  # c_pbest
        social_coefficient: float,  # c_gbest
    ):
        assert jnp.all(lb[0] == lb) and jnp.all(
            ub[0] == ub
        ), "Currently the coevolution framewrok restricts that the upper/lower bound should be the same across dimensions"
        pso = PSO(
            lb[:1],
            ub[:1],
            subpop_size,
            inertia_weight,
            cognitive_coefficient,
            social_coefficient,
        )
        super().__init__(
            base_algorithm=pso,
            dim=lb.shape[0],
            num_subpops=lb.shape[0],
            subpop_size=subpop_size,
            num_subpop_iter=1,
            random_subpop=False,
        )

# --------------------------------------------------------------------------------------
# 1. This code implements algorithms described in the following papers:
#
# Title: Small worlds and mega-minds: effects of neighborhood topology on particle swarm performance
# Link: https://ieeexplore.ieee.org/document/785509
#
# Title: Population structure and particle swarm performance
# Link: https://ieeexplore.ieee.org/document/1004493
# --------------------------------------------------------------------------------------


import jax
import jax.numpy as jnp
from typing import Literal
from .topology_utils import (
    get_circles_neighbour,
    get_neighbour_best_fitness,
    build_adjacancy_list_from_matrix,
)

from evox import Algorithm, State


class SwmmPSO(Algorithm):
    def __init__(
        self,
        lb,
        ub,
        pop_size,
        max_phi_1=2.05,
        max_phi_2=2.05,
        max_phi=4.1,
        mean=None,
        stdev=None,
        topology: Literal["Circles", "Wheels", "Stars", "Random"] = "Circles",
        shortcut: int = 0,
    ):
        self.dim = lb.shape[0]
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        self.max_phi_1 = max_phi_1
        self.max_phi_2 = max_phi_2
        self.max_phi = max_phi
        self.mean = mean
        self.stdev = stdev
        self.topology = topology
        self.shortcut = shortcut

    """
        PSO uses the version from "The particle swarm - explosion, stability, and convergence in a multidimensional complex space"
    """

    def setup(self, key):
        state_key, init_pop_key, init_v_key, adj_key = jax.random.split(key, 4)
        if self.mean is not None and self.stdev is not None:
            population = self.stdev * jax.random.normal(
                init_pop_key, shape=(self.pop_size, self.dim)
            )
            population = jnp.clip(population, self.lb, self.ub)
            velocity = self.stdev * jax.random.normal(
                init_v_key, shape=(self.pop_size, self.dim)
            )
        else:
            length = self.ub - self.lb
            population = jax.random.uniform(
                init_pop_key, shape=(self.pop_size, self.dim)
            )
            population = population * length + self.lb
            velocity = jax.random.uniform(init_v_key, shape=(self.pop_size, self.dim))
            velocity = velocity * length * 2 - length

        if self.max_phi > 0:
            phi = jnp.ones(shape=(self.pop_size, 1)) * self.max_phi
        else:
            phi = jnp.ones(shape=(self.pop_size, 1)) * (self.max_phi_1 + self.max_phi_2)

        # equation in original paper where chi = 1-1/phi+\sqrt{\abs{phi**2-4*phi}} result in wrong coefficients
        # --> chi = 1 - 1 / phi + jnp.sqrt(jnp.abs(phi * (phi - 4))) / 2
        # the proper coefficient are chi = 2/(phi - 2 + jnp.sqrt(jnp.abs(phi * (phi - 4))))
        chi = 2 / (phi - 2 + jnp.sqrt(jnp.abs(phi * (phi - 4))))

        adjacancy_matrix = self._get_topo(key=adj_key, population=population)

        return State(
            population=population,
            velocity=velocity,
            local_best_location=population,
            local_best_fitness=jnp.full((self.pop_size,), jnp.inf),
            neighbour_best_location=population,
            neighbour_best_fitness=jnp.full((self.pop_size,), jnp.inf),
            adjacancy_matrix=adjacancy_matrix,
            key=state_key,
            chi=chi,
            phi=phi,
        )

    def _get_topo(self, key, population):
        adjacancy_matrix: jax.Array
        if self.topology == "Circles":
            adjacancy_matrix = get_circles_neighbour(
                random_key=key, population=population, K=2, shortcut=self.shortcut
            )
        return adjacancy_matrix

    def ask(self, state):
        return state.population, state

    def tell(self, state, fitness):
        key, key1, key2, key3 = jax.random.split(state.key, 4)

        # is it same for all or different
        phi1 = jax.random.uniform(
            key1, shape=(self.pop_size, self.dim), minval=0, maxval=self.max_phi_1
        )
        phi2 = jax.random.uniform(
            key2, shape=(self.pop_size, self.dim), minval=0, maxval=self.max_phi_2
        )

        compare = state.local_best_fitness > fitness
        local_best_location = jnp.where(
            compare[:, jnp.newaxis], state.population, state.local_best_location
        )
        local_best_fitness = jnp.minimum(state.local_best_fitness, fitness)

        # adjacancy_matrix = self._get_topo(key=key3, population=state.population)
        adjacancy_matrix = state.adjacancy_matrix

        neighbour_list, _ = build_adjacancy_list_from_matrix(
            adjacancy_matrix=adjacancy_matrix
        )

        neighbour_best_fitness, neighbour_best_indice = get_neighbour_best_fitness(
            fitness=local_best_fitness, adjacancy_list=neighbour_list
        )

        neighbour_best_location = local_best_location[neighbour_best_indice, :]

        # vi = chi * (vi + phi1 * (pi - xi) + phi2 * (pg - xi))
        # pi is the best position in the search space it has found thus far in a vector
        # g is the index of the particle in the neighborhood with the best performance so far.

        velocity = state.chi * (
            state.velocity
            + phi1 * (local_best_location - state.population)
            + phi2 * (neighbour_best_location - state.population)
        )

        population = state.population + velocity
        population = jnp.clip(population, self.lb, self.ub)

        return state.update(
            population=population,
            velocity=velocity,
            local_best_location=local_best_location,
            local_best_fitness=local_best_fitness,
            neighbour_best_location=neighbour_best_location,
            neighbour_best_fitness=neighbour_best_fitness,
            adjacancy_matrix=adjacancy_matrix,
            key=key,
        )

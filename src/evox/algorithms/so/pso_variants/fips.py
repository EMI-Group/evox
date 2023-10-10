# --------------------------------------------------------------------------------------
# 1. This code implements algorithms described in the following papers:
#
# Title: The fully informed particle swarm: simpler, maybe better
# Link: https://ieeexplore.ieee.org/document/1304843
# --------------------------------------------------------------------------------------

import jax
import jax.numpy as jnp
from .utils import get_distance_matrix
from typing import Literal
from .topology_utils import (
    get_square_neighbour,
    get_full_neighbour,
    build_adjacancy_list_from_matrix,
)
from evox import Algorithm, State


class FIPS(Algorithm):
    def __init__(
        self,
        lb,
        ub,
        pop_size,
        max_phi=4.1,
        mean=None,
        stdev=None,
        topology: Literal[
            "Square", "Ring", "USquare", "URing", "All", "UAll"
        ] = "Square",
        weight_type: Literal["Constant", "Pbest", "Distance"] = "Distance",
        shortcut: int = 0,
    ):
        self.dim = lb.shape[0]
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        self.max_phi = max_phi
        self.mean = mean
        self.stdev = stdev
        self.topology = topology
        self.shortcut = shortcut
        self.weight_type = weight_type

    """
        PSO uses the version from "The particle swarm - explosion, stability, and convergence in a multidimensional complex space"
    """

    def setup(self, key):
        state_key, init_pop_key, init_v_key = jax.random.split(key, 3)
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

        adjacancy_matrix: jax.Array
        if self.topology in ["Square", "USquare"]:
            adjacancy_matrix = get_square_neighbour(population=population)
        elif self.topology in ["All", "UAll"]:
            adjacancy_matrix = get_full_neighbour(population=population)
        else:
            raise NotImplementedError()

        phi = jnp.ones(shape=(self.pop_size, 1)) * self.max_phi
        # chi = 1-1/phi+\sqrt{\abs{phi**2-4*phi}}
        chi = 2 / (phi - 2 + jnp.sqrt(jnp.abs(phi * (phi - 4))))

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

    def ask(self, state):
        return state.population, state

    def tell(self, state, fitness):
        key, key1 = jax.random.split(state.key, 2)

        compare = state.local_best_fitness > fitness
        local_best_location = jnp.where(
            compare[:, jnp.newaxis], state.population, state.local_best_location
        )
        local_best_fitness = jnp.minimum(state.local_best_fitness, fitness)

        adjacancy_matrix = state.adjacancy_matrix

        neighbour_list, neighbour_list_masking = build_adjacancy_list_from_matrix(
            adjacancy_matrix=adjacancy_matrix
        )

        # jax.debug.print("neighbour {}", neighbour_list)

        # vi = chi * (vi + phi(pm - xi))
        # pm is calculated using method proposed in paper

        weight: jax.Array
        if self.weight_type == "Constant":
            weight = self._calculate_weight_by_constant(adjacancy_list=neighbour_list)
        elif self.weight_type == "Pbest":
            weight = self._calculate_weight_by_fitness(
                fitness=local_best_fitness, adjacancy_list=neighbour_list
            )
        elif self.weight_type == "Distance":
            weight = self._calculate_weight_by_distance(
                location=local_best_location, adjacancy_list=neighbour_list
            )

        calculated_pm = self._get_PM(
            weight_list=weight,
            adjacancy_list=neighbour_list,
            adjacancy_list_mapping=neighbour_list_masking,
            location=local_best_location,
            key=key1,
        )

        velocity = state.chi * (
            state.velocity + state.phi * (calculated_pm - state.population)
        )

        population = state.population + velocity
        population = jnp.clip(population, self.lb, self.ub)

        return state.update(
            population=population,
            velocity=velocity,
            local_best_location=local_best_location,
            local_best_fitness=local_best_fitness,
            key=key,
        )

    def _get_PM(
        self, weight_list, adjacancy_list, adjacancy_list_mapping, location, key
    ):
        phik = jax.random.uniform(
            key=key, shape=(self.pop_size, self.pop_size, self.dim)
        )
        phik = adjacancy_list_mapping[:, :, jnp.newaxis] * phik * self.max_phi
        weight_phi = weight_list[:, :, jnp.newaxis] * phik

        def calculate_pm(row_weight, row_adjacancy_list):
            upper = location[row_adjacancy_list] * row_weight
            lower = row_weight

            upper = jnp.sum(upper, axis=0)
            lower = jnp.sum(lower, axis=0)

            frac = upper / lower
            return frac.reshape(-1)

        result = jax.vmap(calculate_pm, in_axes=0)(weight_phi, adjacancy_list)
        return result

    def _calculate_weight_by_constant(self, adjacancy_list):
        return jnp.ones_like(adjacancy_list)

    def _calculate_weight_by_fitness(self, fitness, adjacancy_list):
        """
        each neighbor was weighted by the goodness of its previous best;
        goodness is set as 1/fitness.
        """
        weight = 1 / fitness[adjacancy_list]
        return weight

    def _calculate_weight_by_distance(self, location, adjacancy_list):
        N = adjacancy_list.shape[0]
        distance_matrix = get_distance_matrix(location)
        # print(adjacancy_list.dtype)
        row_indice = jnp.arange(N, dtype=adjacancy_list.dtype)

        def get_row_distance(neighbour, indice):
            row_distance = distance_matrix[indice, neighbour]
            return row_distance

        distance_list = jax.vmap(get_row_distance, in_axes=0)(
            adjacancy_list, row_indice
        )

        return distance_list

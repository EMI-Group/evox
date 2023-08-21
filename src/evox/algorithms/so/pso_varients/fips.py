"""
The Fully Informed Particle Swarm: Simpler, Maybe Better

Not fully tested.
"""
import jax
import jax.numpy as jnp
from evox.algorithms.so.pso_varients.utils import row_argsort, get_distance_matrix, select_from_mask
from functools import partial
from typing import Union, Iterable, Literal
from evox.algorithms.so.pso_varients.topology_utils import (
    get_square_neighbour, 
    get_neighbour_best_fitness, 
    get_full_neighbour,
    build_adjacancy_list_from_matrix
)


from evox import (
    Algorithm,
    Problem,
    State,
    algorithms,
    jit_class,
    monitors,
    pipelines,
    problems,
)

from evox.utils import min_by


class FIPS(Algorithm):
    def __init__(
        self,
        lb,
        ub,
        pop_size,
        max_phi = 4.1,
        mean=None,
        stdev=None,
        topology: Literal['Square', 'Ring', "USquare", 'URing', "All", "UAll"] = "Square" ,
        weight_type: Literal['Constant', 'Pbest', "Distance"] = "Distance",
        shortcut: int = 0
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


        adjacancy_matrix : jax.Array
        if self.topology in ["Square", "USquare"]:
            adjacancy_matrix = get_square_neighbour(population = population)
        elif self.topology in ["All", "UAll"]:
            adjacancy_matrix = get_full_neighbour(population = population)
        else:
            raise NotImplementedError()
        
        phi = jnp.ones(shape=(self.pop_size, 1)) * self.max_phi
        # chi = 1-1/phi+\sqrt{\abs{phi**2-4*phi}}
        chi = 2/(phi - 2 + jnp.sqrt(jnp.abs(phi * (phi - 4))))

        return State(
            population=population,
            velocity=velocity,
            local_best_location=population,
            local_best_fitness=jnp.full((self.pop_size,), jnp.inf),
            neighbour_best_location=population,
            neighbour_best_fitness=jnp.full((self.pop_size,), jnp.inf),
            adjacancy_matrix = adjacancy_matrix,
            key=state_key,
            chi=chi,
            phi=phi
        )

    def ask(self, state):
        return state.population, state

    def tell(self, state, fitness):
        key, key1= jax.random.split(state.key, 2)


        compare = state.local_best_fitness > fitness
        local_best_location = jnp.where(
            compare[:, jnp.newaxis], state.population, state.local_best_location
        )
        local_best_fitness = jnp.minimum(state.local_best_fitness, fitness)

        adjacancy_matrix = state.adjacancy_matrix

        

        neighbour_list, neighbour_list_masking = build_adjacancy_list_from_matrix(adjacancy_matrix = adjacancy_matrix)

        # jax.debug.print("neighbour {}", neighbour_list)

        # vi = chi * (vi + phi(pm - xi))
        # pm is calculated using method proposed in paper

        weight: jax.Array
        if self.weight_type == "Constant":
            weight = self._calculate_weight_by_constant(adjacancy_list=neighbour_list)
        elif self.weight_type == "Pbest":
            weight = self._calculate_weight_by_fitness(fitness=local_best_fitness, adjacancy_list=neighbour_list)
        elif self.weight_type == "Distance":
            weight = self._calculate_weight_by_distance(location=local_best_location, adjacancy_list=neighbour_list)

        calculated_pm = self._get_PM(weight_list = weight, 
                                     adjacancy_list = neighbour_list, 
                                     adjacancy_list_mapping = neighbour_list_masking, 
                                     location = local_best_location, key = key1)

        velocity = state.chi * ( state.velocity
            + state.phi * (calculated_pm - state.population)
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

    def _get_PM(self, weight_list, adjacancy_list, adjacancy_list_mapping, location, key):
        phik = jax.random.uniform(key = key, shape=adjacancy_list.shape)
        phik = adjacancy_list_mapping * phik * self.max_phi
        weight_phi = weight_list * phik

        def calculate_pm(row_weight,row_adjacancy_list ):
            upper = location[row_adjacancy_list] * row_weight[jnp.newaxis,:,jnp.newaxis]
            lower = row_weight[jnp.newaxis,:,jnp.newaxis]

            upper = jnp.sum(upper, axis=1)
            lower = jnp.sum(lower)

            frac = upper/lower
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
        weight = 1/fitness[adjacancy_list]
        return weight


    def _calculate_weight_by_distance(self, location, adjacancy_list):
        N = adjacancy_list.shape[0]
        distance_matrix = get_distance_matrix(location)
        # print(adjacancy_list.dtype)
        row_indice = jnp.arange(N, dtype=adjacancy_list.dtype)
        
        def get_row_distance(neighbour, indice):
            row_distance = distance_matrix[indice, neighbour]
            return row_distance
            
        distance_list = jax.vmap(get_row_distance, in_axes=0)(adjacancy_list, row_indice)

        return distance_list

def test_weight_calculation():
    fitness = jnp.array([7,8,9,10,11])
    adjacancy_list = jnp.array([[0,2,3,0],
                                [1,2,1,1],
                                [0,1,2,3],
                                [0,2,3,3]])
    
    result = jax.jit(FIPS._calculate_weight_by_fitness)(None, fitness=fitness, adjacancy_list=adjacancy_list)
    print(result)
    
    
    key = jax.random.PRNGKey(12)
    values = jax.random.uniform(key, shape=(4, 2))

    result = jax.jit(FIPS._calculate_weight_by_distance)(None, position=values, adjacancy_list=adjacancy_list)
    print(result)
    pass

def test_indexing():
    adjacancy_list = [[0,2,1]]
    location = [[1, 9], [2, 8], [3, 7]]
    adjacancy_list = jnp.array(adjacancy_list)
    location = jnp.array(location)
    print(location[adjacancy_list])
    
    weight = jnp.array([0.1,0.001, 4])
    upper = location[adjacancy_list] * weight[jnp.newaxis,:,jnp.newaxis]
    lower = weight
    print(upper)
    print(lower)

    upper = jnp.sum(upper, axis=1)
    print(upper)
    lower = jnp.sum(lower)
    print(lower)
    frac = upper/lower
    print(frac)
    print(frac.reshape(-1))


if __name__ == "__main__":
    # test_weight_calculation()
    test_indexing()








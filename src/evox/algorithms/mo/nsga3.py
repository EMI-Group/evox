import evox as ex
import jax
import jax.numpy as jnp

from evox.operators.selection import UniformRandomSelection
from evox.operators.mutation import GaussianMutation, PmMutation
from evox.operators.crossover import UniformCrossover, SimulatedBinaryCrossover
from evox.operators.sampling import UniformSampling
from evox.operators import non_dominated_sort, crowding_distance_sort


@ex.jit_class
class NSGA2(ex.Algorithm):
    """NSGA-II algorithm

    link: https://ieeexplore.ieee.org/document/6600851
    """

    def __init__(
        self,
        lb,
        ub,
        n_objs,
        pop_size,
        ref=None,
        selection=UniformRandomSelection(p=0.5),
        mutation=GaussianMutation(),
        # mutation=PmMutation(),
        crossover=UniformCrossover(),
        # crossover=SimulatedBinaryCrossover(),
    ):
        self.lb = lb
        self.ub = ub
        self.n_objs = n_objs
        self.dim = lb.shape[0]
        self.pop_size = pop_size
        self.ref = ref if ref else UniformSampling(n_objs, pop_size).random()

        self.selection = selection
        self.mutation = mutation
        self.crossover = crossover

    def setup(self, key):
        key, subkey = jax.random.split(key)
        population = (
            jax.random.uniform(subkey, shape=(self.pop_size, self.dim))
            * (self.ub - self.lb)
            + self.lb
        )
        return ex.State(
            population=population,
            fitness=jnp.zeros((self.pop_size, self.n_objs)),
            next_generation=population,
            is_init=True,
        )

    def ask(self, state):
        return jax.lax.cond(state.is_init, self._ask_init, self._ask_normal, state)

    def tell(self, state, fitness):
        return jax.lax.cond(
            state.is_init, self._tell_init, self._tell_normal, state, fitness
        )

    def _ask_init(self, state):
        return state.population, state

    def _ask_normal(self, state):
        mutated, state = self.selection(state, state.population)
        mutated, state = self.mutation(state, mutated)

        crossovered, state = self.selection(state, state.population)
        crossovered, state = self.crossover(state, crossovered)

        next_generation = jnp.clip(
            jnp.concatenate([mutated, crossovered], axis=0), self.lb, self.ub
        )
        return next_generation, state.update(next_generation=next_generation)

    def _tell_init(self, state, fitness):
        state = state.update(fitness=fitness, is_init=False)
        return state

    def _tell_normal(self, state, fitness):
        merged_pop = jnp.concatenate([state.population, state.next_generation], axis=0)
        merged_fitness = jnp.concatenate([state.fitness, fitness], axis=0)

        rank = non_dominated_sort(merged_fitness)
        order = jnp.argsort(rank)
        rank = rank[order]
        ranked_pop = merged_pop[order]
        ranked_fitness = merged_fitness[order]
        last_rank = rank[self.pop_size]
        end = jnp.sum(rank <= last_rank)
        mask = rank <= last_rank
        ranked_fitness = jnp.where(jnp.repeat(mask[:, None], self.n_objs), ranked_fitness, jnp.nan)
        
        # Normalize
        def normalize_loop(i, val):
            ex_idx, fitness = val
            weight = jnp.full(self.n_objs, 1e-6).at[i].set(1)
            asf = fitness / weight
            idx = jnp.nanargmin(jnp.nanmax(asf, axis=1))
            ex_idx = ex_idx.at[i].set(idx)
            return ex_idx, fitness
        
        ideal = jnp.nanmin(ranked_fitness, axis=0)
        ex_idx = jnp.full(self.n_objs, jnp.nan)
        ex_idx, _ = jax.lax.fori_loop(0, self.n_objs, normalize_loop, (ex_idx, ranked_fitness))
        extreme = ranked_fitness[ex_idx]
        
        def extreme_point(val):
            extreme = val[0]
            plane = jnp.linalg.solve(extreme, jnp.ones(self.n_objs))
            intercept = 1/ plane
            return intercept
        
        def worst_point(val):
            return jnp.nanmax(ranked_fitness, axis=0)
        
        nadir_point = jax.lax.cond(jnp.linalg.matrix_rank(extreme) == self.n_objs,
                                   extreme_point, worst_point,
                                   (extreme, ranked_fitness-ideal))
        normalized_fitness = (ranked_fitness - ideal) / nadir_point
        
        # Associate
        def perpendicular_distance(x, y):
            y_norm = jnp.linalg.norm(y, axis=1)
            proj_len = x @ y.T / y_norm
            unit_vec = y / y_norm[:, None]
            proj_vec = jnp.reshape(proj_len, (proj_len.size, 1)) * jnp.tile(unit_vec, (len(x), 1))
            prep_vec = jnp.repeat(x, len(y), axis=0) - proj_vec
            dist = jnp.reshape(jnp.linalg.norm(prep_vec, axis=1), (len(x), len(y)))
            return dist
        
        dist = perpendicular_distance(ranked_fitness, self.ref)
        pi = jnp.nanargmin(dist, axis=1)
        d = dist[jnp.arange(len(normalized_fitness)), pi]
    
        # Niche
        def niche_loop(val):
            sur, sur_fit, pop, fit, idx = val
            
            return NotImplemented
        
        survivor = jnp.full((self.pop_size, self.dim), jnp.nan)
        survivor_fitness = jnp.full((self.pop_size, self.n_objs), jnp.nan)
        K = self.pop_size - jnp.sum(rank < last_rank)
        
        
        state = state.update(population=survivor, fitness=survivor_fitness)
        return state

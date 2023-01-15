import evox as ex
import jax
import jax.numpy as jnp

from evox.operators.selection import ReferenceVectorGuidedSelection
from evox.operators.mutation import PmMutation
from evox.operators.crossover import SimulatedBinaryCrossover
from evox.operators.sampling import UniformSampling, LatinHypercubeSampling
    
    
@ex.jit_class
class RVEA(ex.Algorithm):
    """RVEA algorithms

    link: https://ieeexplore.ieee.org/document/7386636
    """

    def __init__(
        self,
        lb,
        ub,
        n_objs,
        pop_size,
        alpha=2,
        fr=0.1,
        max_gen=100,
        selection=ReferenceVectorGuidedSelection(),
        mutation=PmMutation(),
        crossover=SimulatedBinaryCrossover(),
    ):
        self.lb = lb
        self.ub = ub
        self.n_objs = n_objs
        self.dim = lb.shape[0]
        self.pop_size = pop_size
        self.alpha = alpha
        self.fr = fr
        self.max_gen = max_gen

        self.selection = selection
        self.mutation = mutation
        self.crossover = crossover
        # self.sampling = UniformSampling(self.pop_size, self.n_objs)
        self.sampling = LatinHypercubeSampling(self.pop_size, self.n_objs)

    def setup(self, key):
        key, subkey1, subkey2 = jax.random.split(key, 3)
        population = (
            jax.random.uniform(subkey1, shape=(self.pop_size, self.dim))
            * (self.ub - self.lb)
            + self.lb
        )
        v = self.sampling.random(subkey2)[0]
        v = v / jnp.linalg.norm(v, axis=0)
        mask = jnp.full((self.pop_size, 1), False)

        return ex.State(
            population=population,
            fitness=jnp.zeros((self.pop_size, self.n_objs)),
            next_generation=population,
            reference_vector=v,
            is_init=True, key=key, gen=0, mask=mask)

    @ex.jit_method
    def ask(self, state):
        return jax.lax.cond(state.is_init, self._ask_init, self._ask_normal, state)

    def tell(self, state, fitness):
        return jax.lax.cond(state.is_init,
                            self._tell_init,
                            self._tell_normal,
                            state, fitness
                            )

    def _ask_init(self, state):
        return state, state.population

    def _ask_normal(self, state):
        mask = state.mask
        key = state.key
        key, subkey = jax.random.split(key)
        r = jax.random.uniform(subkey, shape=(self.pop_size, self.dim))* (self.ub - self.lb) + self.lb
        
        state, crossovered = self.crossover(state, state.population)
        state, next_generation = self.mutation(state, crossovered, (self.lb, self.ub))
        next_generation = jnp.where(mask, r, next_generation)

        return state.update(next_generation=next_generation, key=key), next_generation

    def _tell_init(self, state, fitness):
        state = state.update(fitness=fitness, is_init=False)
        return state

    def _tell_normal(self, state, fitness):
        current_gen = state.gen + 1
        v = state.reference_vector
        merged_pop = jnp.concatenate([state.population, state.next_generation], axis=0)
        merged_fitness = jnp.concatenate([state.fitness, fitness], axis=0)

        state, rank = self.selection(state, merged_fitness, v, (current_gen/self.max_gen)**self.alpha)

        mask = (rank == -1)[:, jnp.newaxis]

        survivor = merged_pop[rank]
        survivor_fitness = merged_fitness[rank]
        
        def rv_adaptation(pop_obj, v):
            v_temp = v * jnp.tile((pop_obj.max(0) - pop_obj.min(0)), (len(v), 1))
            next_v = v.astype(float)

            next_v = v_temp / jnp.tile(jnp.sqrt(jnp.sum(v_temp**2, axis=1)).reshape(len(v), 1), (1, jnp.shape(v)[1]))
                
            return next_v
        
        def no_update(pop_obj, v):
            return v
        
        v= jax.lax.cond(current_gen % (1 / self.fr) == 0, rv_adaptation, no_update, survivor_fitness, v)

        state = state.update(population=survivor, fitness=survivor_fitness, reference_vector=v, gen=current_gen, mask=mask)
        return state

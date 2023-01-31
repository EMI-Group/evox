import evox as ex
import jax
import jax.numpy as jnp

from evox.operators.selection import TournamentSelection
from evox.operators.mutation import PmMutation
from evox.operators.crossover import SimulatedBinaryCrossover
from evox.utils import cal_fitness


@ex.jit_class
class IBEA(ex.Algorithm):
    """IBEA algorithm

    link: 
    """

    def __init__(
        self,
        lb,
        ub,
        n_objs,
        pop_size,
        kappa=0.05,
        mutation=PmMutation(),
        crossover=SimulatedBinaryCrossover(),
    ):
        self.lb = lb
        self.ub = ub
        self.n_objs = n_objs
        self.dim = lb.shape[0]
        self.pop_size = pop_size
        self.kappa = kappa

        self.selection = TournamentSelection(num_round=self.pop_size)
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
            is_init=True)

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
        population = state.population
        pop_obj = state.fitness
        fitness = cal_fitness(pop_obj, self.kappa)[0]

        state, selected = self.selection(state, population, fitness)
        state, crossovered = self.crossover(state, selected)
        state, mutated = self.mutation(state, crossovered, (self.lb, self.ub))

        next_generation = jnp.clip(mutated, self.lb, self.ub)
        return state.update(next_generation=next_generation), next_generation

    def _tell_init(self, state, fitness):
        state = state.update(fitness=fitness, is_init=False)
        return state

    #@profile
    def _tell_normal(self, state, fitness):
        merged_pop = jnp.concatenate(
            [state.population, state.next_generation], axis=0)
        merged_obj = jnp.concatenate([state.fitness, fitness], axis=0)

        n = jnp.shape(merged_pop)[0]
        merged_fitness, I, C = cal_fitness(merged_obj, self.kappa)

        next_ind = jnp.arange(n)
        vals = (next_ind, merged_fitness)
        
        def body_fun(i, vals):
            next_ind, merged_fitness = vals
            x = jnp.argmin(merged_fitness)
            merged_fitness += jnp.exp(-I[x, :] / C[x] / self.kappa)
            merged_fitness = merged_fitness.at[x].set(jnp.max(merged_fitness))
            next_ind = next_ind.at[x].set(-1)
            return (next_ind, merged_fitness)
        
        next_ind, merged_fitness = jax.lax.fori_loop(0, self.pop_size, body_fun, vals)

        ind = jnp.where(next_ind != -1, size=n, fill_value=-1)[0]
        ind_n = ind[0:self.pop_size]

        survivor = merged_pop[ind_n]
        survivor_fitness = merged_obj[ind_n]

        state = state.update(population=survivor, fitness=survivor_fitness)

        return state

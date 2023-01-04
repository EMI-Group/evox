import jax
import jax.numpy as jnp
import evox
from evox import State, Algorithm


class NaiveES(Algorithm):
    def __init__(self, dim, pop_size, topk):
        self.dim = dim
        self.pop_size = pop_size
        self.topk = topk

    def setup(self, key):
        mean = jnp.zeros((dim,))
        stdvar = jnp.ones((self.dim,))
        return State(
            mean=mean,
            stdvar=stdvar,
            key=key
        )

    def ask(self, state):
        key, subkey = jax.random.split(state.key)
        noise = jax.random.normal(
            subkey,
            (self.pop_size, self.dim) 
        )
        pop = state.mean + state.stdvar * noise
        new_state = state.update(
            key=key,
            pop=pop
        )
        return new_state, sample

    def tell(self, state, fitness):
        _topk_value, topk_index = jax.lax.top_k(
            fitness,
            self.topk
        )
        elite = state.pop[topk_index]
        new_mean = jnp.mean(elite, axis=0)
        new_stdvar = jnp.std(elite, axis=0)
        new_state = state.update(
            mean=new_mean,
            stdvar=new_stdvar
        )
        return new_state

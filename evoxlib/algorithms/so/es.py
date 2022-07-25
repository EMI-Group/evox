import jax
import jax.numpy as jnp
import evoxlib as exl

@exl.use_state_class
class ES(exl.Algorithm):
    def __init__(self, lb, ub, pop_size, topk=None, eps=1e-8):
        self.dim = lb.shape[0]
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        if topk is None:
            self.topk = self.pop_size // 2
        else:
            self.topk = topk
        self.eps = eps

    def setup(self, key):
        state_key, init_key = jax.random.split(key)
        mean = jax.random.uniform(init_key, shape=(self.dim, )) * (self.ub - self.lb) + self.lb
        stdvar = jnp.ones((self.dim, ))
        return {
            'mean': mean,
            'stdvar': stdvar,
            'key': state_key
        }

    def tell(self, state, X, F):
        order = jnp.argsort(F)
        X = X[order]
        elite = X[:self.topk, :]
        new_mean = jnp.mean(elite, axis=0)
        new_stdvar = jnp.sqrt(jnp.var(elite, axis=0) + self.eps)
        return state | {
            'mean': new_mean,
            'stdvar': new_stdvar
        }

    def ask(self, state):
        key = state['key']
        key, subkey = jax.random.split(key)
        sample = jax.random.normal(subkey, shape=(self.pop_size, self.dim)) * state['stdvar'] + state['mean']
        sample = jnp.clip(sample, self.lb, self.ub)
        state = state | {'key': key}
        return state, sample
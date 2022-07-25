import chex
import jax

def jit(func):
    return jax.jit(func, static_argnums=(0,))

def lift(func):
    def wrapper(self, state, *args, **kargs):
        if self.name == '_top_level':
            return func(self, state, *args, **kargs)
        else:
            return state | {
                self.name: func(self, state[self.name], *args, **kargs)
            }
    return wrapper

class Module:
    def setup(self, key: chex.PRNGKey = None):
        pass

    def init(self, key: chex.PRNGKey = None, name='_top_level'):
        self.name = name
        state = {}
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Module):
                key, subkey = jax.random.split(key)
                submodule_name = f'_submodule_{attr_name}'
                state[submodule_name] = attr.init(subkey, submodule_name)
        return self.setup(key) | state
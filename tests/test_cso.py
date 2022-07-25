import chex
import evoxlib as exl
import jax
import jax.numpy as jnp
import pytest

@exl.jit_class
@exl.use_state_class
class Pipeline(exl.Module):
    def __init__(self):
        self.cso = exl.algorithms.CSO(
            lb=jnp.full(shape=(2, ), fill_value=-32),
            ub=jnp.full(shape=(2, ), fill_value=32),
            pop_size=100
        )
        self.ackley = exl.problems.classic.Ackley()

    def setup(self, key):
        return {
            'min_fitness': 1e9
        }

    def step(self, state):
        state, X = self.cso.ask(state)
        state, F = self.ackley.evaluate(state, X)
        state = self.cso.tell(state, X, F)
        return state | {
            'min_fitness': jnp.minimum(state['min_fitness'], jnp.min(F))
        }

    def get_min_fitness(self, state):
        return state, state['min_fitness']

def test_cso():
    pipeline = Pipeline()
    key = jax.random.PRNGKey(42)
    state = pipeline.init(key)
    for i in range(100):
        state = pipeline.step(state)
    state, min_fitness = pipeline.get_min_fitness(state)
    assert min_fitness < 1e-4
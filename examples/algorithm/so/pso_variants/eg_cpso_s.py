from evox import algorithms, problems, pipelines, monitors
import jax
import jax.numpy as jnp
from functools import partial
import evox as ex

algorithm = algorithms.so.pso_vatients.CPSO_S(
    lb=jnp.full(shape=(10,), fill_value=-32),
    ub=jnp.full(shape=(10,), fill_value=32),
    pop_size=15,
    inertia_weight=0.4,
    pbest_coefficient=2.5,
    gbest_coefficient=0.8,
)

def _ackley_func(a, b, c, x):
    return (
        -a * jnp.exp(-b * jnp.sqrt(jnp.mean(x**2)))
        - jnp.exp(jnp.mean(jnp.cos(c * x)))
        + a
        + jnp.e
    )

@ex.jit_class
class Ackley(ex.Problem):
    def __init__(self, a=20, b=0.2, c=2*jnp.pi):
        self.a = a
        self.b = b
        self.c = c

    def evaluate(self, state, X):
        return jax.vmap(jax.vmap(partial(_ackley_func, self.a, self.b, self.c)))(X), state
    
problem = Ackley()

monitor = monitors.FitnessMonitor()

# create a pipeline

pipeline = pipelines.StdPipeline(
    algorithm=algorithm,
    problem=problem,
    fitness_transform=monitor.update,
)

# init the pipeline
key = jax.random.PRNGKey(42)
state = pipeline.init(key)

# run the pipeline for 100 steps
for i in range(100):
    state = pipeline.step(state)
    print(monitor.get_min_fitness())
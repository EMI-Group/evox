import evoxlib as exl
import jax
import jax.numpy as jnp


class Pipeline(exl.Module):
    def __init__(self):
        # choose an algorithm
        self.algorithm = exl.algorithms.CSO(
            lb=jnp.full(shape=(2,), fill_value=-32),
            ub=jnp.full(shape=(2,), fill_value=32),
            pop_size=100,
        )
        # choose a problem
        self.problem = exl.problems.classic.Ackley()
        self.monitor = exl.monitors.FitnessMonitor()
        self.pop_monitor = exl.monitors.PopulationMonitor(2)

    def step(self, state):
        # one step
        state, X = self.algorithm.ask(state)
        self.pop_monitor.update(X)
        state, F = self.problem.evaluate(state, X)
        self.monitor.update(F)
        state = self.algorithm.tell(state, X, F)
        return state


# create a pipeline
pipeline = Pipeline()
# init the pipeline
key = jax.random.PRNGKey(42)
state = pipeline.init(key)

# run the pipeline for 100 steps
for i in range(100):
    state = pipeline.step(state)

pipeline.monitor.show()
pipeline.pop_monitor.show()

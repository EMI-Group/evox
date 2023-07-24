from evox import algorithms, problems, pipelines, monitors
import jax
import jax.numpy as jnp

algorithm = algorithms.so.pso_vatients.CLPSO(
    lb=jnp.full(shape=(10,), fill_value=-32),
    ub=jnp.full(shape=(10,), fill_value=32),
    pop_size=100,
    inertia_weight=0.4,
    const_coefficient=0.6,
    learning_probability=jnp.full(shape=(100,), fill_value=0.5),
)

problem = problems.classic.Ackley()

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
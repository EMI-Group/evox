from evox import algorithms, problems, workflows
import jax
import jax.numpy as jnp

algorithm = algorithms.PSO(
    lb=jnp.full(shape=(2,), fill_value=-32),
    ub=jnp.full(shape=(2,), fill_value=32),
    pop_size=100,
)

problem = problems.numerical.Ackley()

# create a workflow

workflow = workflows.StdWorkflow(
    algorithm,
    problem,
)

# init the workflow
key = jax.random.PRNGKey(42)
state = workflow.init(key)

# run the workflow for 100 steps
for i in range(100):
    state = workflow.step(state)

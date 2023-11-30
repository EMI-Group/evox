from evox import workflows, algorithms, problems
from evox.monitors import StdSOMonitor
from evox.utils import TreeAndVector
import jax
import jax.numpy as jnp
from flax import linen as nn
import pytest


@pytest.mark.parametrize("batch_policy", [True, False])
def test_cartpole(batch_policy):
    class CartpolePolicy(nn.Module):
        """A simple model for cartpole"""

        @nn.compact
        def __call__(self, x):
            x = nn.Dense(32)(x)
            x = nn.sigmoid(x)
            x = nn.Dense(2)(x)
            return jnp.argmax(x)

    key = jax.random.PRNGKey(42)
    model_key, workflow_key = jax.random.split(key)

    model = CartpolePolicy()
    params = model.init(model_key, jnp.zeros((4,)))
    adapter = TreeAndVector(params)
    monitor = StdSOMonitor()
    problem = problems.neuroevolution.Gym(
        env_name="CartPole-v1",
        policy=jax.jit(model.apply),
        num_workers=3,
        worker_options={"num_gpus": 0, "num_cpus": 0},
        controller_options={
            "num_cpus": 0,
            "num_gpus": 0,
        },
        batch_policy=batch_policy,
    )
    center = adapter.to_vector(params)
    # create a workflow
    workflow = workflows.UniWorkflow(
        algorithm=algorithms.CSO(
            lb=jnp.full_like(center, -10.0),
            ub=jnp.full_like(center, 10.0),
            mean=center,
            stdev=0.1,
            pop_size=16,
        ),
        problem=problem,
        monitor=monitor,
        jit_problem=False,
        num_objectives=1,
        pop_transform=adapter.batched_to_tree,
        opt_direction="max",
    )
    # init the workflow
    state = workflow.init(workflow_key)

    # run the workflow for 2 steps
    for i in range(2):
        state = workflow.step(state)

    monitor.flush()
    min_fitness = monitor.get_best_fitness()
    # gym is deterministic, so the result should always be the same
    assert min_fitness == 40.0

    # run the workflow for another 25 steps
    for i in range(25):
        state = workflow.step(state)

    monitor.flush()
    min_fitness = monitor.get_best_fitness()
    assert min_fitness == 48.0
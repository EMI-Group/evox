from evox import workflows, algorithms, problems, use_state
from evox.monitors import EvalMonitor
from evox.utils import TreeAndVector
import jax
import jax.numpy as jnp
from flax import linen as nn
import pytest


def test_envpool_cartpole():
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
    monitor = EvalMonitor()
    problem = problems.neuroevolution.EnvPool(
        env_name="CartPole-v1",
        num_envs=16,
        policy=jax.jit(model.apply),
    )
    center = adapter.to_vector(params)
    # create a workflow
    workflow = workflows.StdWorkflow(
        algorithm=algorithms.PGPE(
            optimizer="adam",
            center_init=center,
            pop_size=16,
        ),
        problem=problem,
        monitors=[monitor],
        external_problem=True,
        num_objectives=1,
        candidate_transforms=[adapter.batched_to_tree],
        opt_direction="max",
    )
    # init the workflow
    state = workflow.init(workflow_key)

    # run the workflow for 10 steps
    for i in range(5):
        state = workflow.step(state)

    min_fitness, state = use_state(monitor.get_best_fitness)(state)
    # envpool is deterministic, so the result should always be the same
    assert min_fitness == 59.0

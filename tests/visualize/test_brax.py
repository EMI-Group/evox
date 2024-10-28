from evox import problems
from evox.utils import frames2gif
import jax
import jax.numpy as jnp
import pytest

gym_name = "ant"  # choose a setup


def random_policy(rand_seed, x):  # weights, observation
    return jnp.tanh(
        jax.random.normal(
            jax.random.PRNGKey(jnp.array(x[0] * 1e7, dtype=jnp.int32) + rand_seed),
            shape=(8,),
        )
    )


def random_stateful_policy(state, rand_seed, x):  # state, weights, observation
    return jnp.tanh(
        jax.random.normal(
            jax.random.PRNGKey(jnp.array(x[0] * 1e7, dtype=jnp.int32) + rand_seed),
            shape=(8,),
        )
    ), state


@pytest.mark.parametrize("stateful_policy", [False, True])
def test_brax(stateful_policy):
    seed = 41
    key = jax.random.PRNGKey(seed)

    if stateful_policy:
        policy = random_stateful_policy
    else:
        policy = random_policy

    problem = problems.neuroevolution.Brax(
        env_name=gym_name,
        policy=policy,
        num_episodes=1,
        max_episode_length=3,
        stateful_policy=stateful_policy,
        initial_state=jnp.zeros(10) if stateful_policy else None,
    )

    state = problem.init(key)
    problem.evaluate(state, jnp.arange(3))

    problem.visualize(key, seed, output_type="HTML")
    assert True

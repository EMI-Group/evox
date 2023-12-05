from evox import problems
from evox.utils import frames2gif
import jax
import jax.numpy as jnp
import pytest

gym_name = "CartPole-v1"  # choose a setup


def random_policy(rand_seed, x):  # weights, observation
    return jax.random.randint(
        jax.random.PRNGKey(jnp.array(x[0] * 1e7, dtype=jnp.int32) + rand_seed),
        shape=(),
        minval=0,
        maxval=2,
    )


@pytest.mark.skip(
    reason="cost too much time"
)
def test():
    seed = 41
    key = jax.random.PRNGKey(seed)

    problem = problems.neuroevolution.Gym(
        env_name=gym_name,
        policy=jax.jit(random_policy),
        num_workers=0,
    )

    state = problem.init(key)

    frames, state = problem.visualize(state, key, seed)
    frames2gif(frames, f"{gym_name}_{seed}.gif")
    assert True

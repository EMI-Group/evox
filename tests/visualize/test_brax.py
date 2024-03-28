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


@pytest.mark.skip(
    reason="cost too much time"
)
def test():
    seed = 41
    key = jax.random.PRNGKey(seed)

    #  It takes too much time to render 500 frames (474s on Nvidia RTX 3090)
    #  I think it is good to add a progress bar to shrink waiting experience.
    problem = problems.neuroevolution.Brax(
        env_name=gym_name,
        policy=jax.jit(random_policy),
        cap_episode=500,
    )

    state = problem.init(key)
    frames = problem.visualize(key, seed, output_type="rgb_array", width=250, height=250)
    frames2gif(frames, f"{gym_name}_{seed}.gif")
    assert True

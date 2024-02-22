import jax.numpy as jnp
from evox.operators.gaussian_process import GPRegression
import optax as ox

try:
    from gpjax.likelihoods import Gaussian
except ImportError as e:
    original_error_msg = str(e)

    def Gaussian(*args, **kwargs):
        raise ImportError(
            f'Gaussian requires gpjax, but got "{original_error_msg}" when importing'
        )


def test_gp():
    i = 0
    x = jnp.arange(i + 5)[:, jnp.newaxis]
    pre_x = jnp.array([4, 5, 6, jnp.inf])[:, jnp.newaxis]
    y = (jnp.arange(i + 5) * 6)[:, jnp.newaxis]
    likelihood = Gaussian(num_datapoints=len(x))
    model = GPRegression(likelihood=likelihood)
    model.fit(x, y, optimzer=ox.sgd(0.001, nesterov=True))
    _, mu, std = model.predict(pre_x)
    assert jnp.abs(mu[1] - 2.90525) < 0.001
    assert jnp.abs(std[1] - 4.80366) < 0.001

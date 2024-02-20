import jax.numpy as jnp
from evox.operators.gaussian_process.regression import GPRegression
from gpjax.likelihoods import Gaussian
import optax as ox


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

from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import jax.random as jr

import optax as ox
import tensorflow_probability.substrates.jax as tfp

import gpjax as gpx

from gpjax.mean_functions import Zero
from evox.operators.gaussian_processes.kernels import RBF
from gpjax.objectives import LogPosteriorDensity

tfd = tfp.distributions
identity_matrix = jnp.eye
key = jr.PRNGKey(123)


class GPClassification:
    def __init__(
        self,
        kernel=RBF(),
        meanfun=Zero(),
        likelihood=None,
        object=LogPosteriorDensity(negative=True),
        key=jr.PRNGKey(123),
    ):
        self.kernel = kernel
        self.mean_fun = meanfun
        self.key = key
        self.prior = gpx.gps.Prior(mean_function=meanfun, kernel=kernel)
        self.likelihood = likelihood
        self.object = object
        self.posterior = self.prior * self.likelihood

    def fit(self, x, y, optimzer=ox.adam):
        self.dataset = gpx.Dataset(X=x, y=y)
        self.object(self.posterior, train_data=self.dataset)
        self.opt_posterior, self.history = gpx.fit(
            model=self.posterior,
            objective=self.object,
            train_data=self.dataset,
            optim=optimzer,
            num_iters=500,
            key=self.key,
        )

    def predict(self, x):
        latent_dist = self.opt_posterior.predict(x, train_data=self.dataset)
        predictive_dist = self.opt_posterior.likelihood(latent_dist)

        predictive_mean = predictive_dist.mean()
        predictive_std = predictive_dist.stddev()
        return predictive_dist, predictive_mean, predictive_std

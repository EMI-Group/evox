# https://docs.jaxgaussianprocesses.com/examples/regression/

from jax import config
from jax import jit
import jax.numpy as jnp
import jax.random as jr
import gpjax as gpx
# from gpjax.kernels import RBF
from gpjax.mean_functions import Zero
# from gpjax.likelihoods import Gaussian
from gpjax.objectives import ConjugateMLL
import jax
import optax as ox
from evox.operators.gaussian_process.kernels import RBF
from evox.operators.gaussian_process.likelihoods import Gaussian
# the demand of gpJAX. Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)
class GPRegression:


    def __init__(self, kernel=RBF(), meanfun=Zero(), likelihood=None, object=ConjugateMLL(negative=True),key=jr.PRNGKey(123)):
        self.kernel = kernel
        self.mean_fun = meanfun
        self.key = key
        self.prior = gpx.gps.Prior(mean_function=meanfun, kernel=kernel)
        self.likelihood = likelihood
        self.object = object
        self.posterior = self.prior * self.likelihood


    def fit(self, x, y, optimzer=ox.GradientTransformation):
        self.dataset = gpx.Dataset(X=x, y=y)
        self.object(self.posterior, train_data=self.dataset)
        # add jit
        self.opt_posterior, self.history = gpx.fit(
            model= self.posterior,
            objective= self.object,
            train_data= self.dataset,
            optim = optimzer,
            num_iters=500,
            key=self.key
        )

    def fit_scipy(self,x,y):
        self.dataset = gpx.Dataset(X=x, y=y)
        self.object(self.posterior, train_data=self.dataset)
        self.opt_posterior, self.history = gpx.fit_scipy(
            model= self.posterior,
            objective= self.object,
            train_data= self.dataset,
        )


    def predict(self,x):
        latent_dist = self.opt_posterior.predict(x, train_data=self.dataset)
        predictive_dist = self.opt_posterior.likelihood(latent_dist)

        predictive_mean = predictive_dist.mean()
        predictive_std = predictive_dist.stddev()
        return predictive_mean, predictive_std

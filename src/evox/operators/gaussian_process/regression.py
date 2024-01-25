# https://docs.jaxgaussianprocesses.com/examples/regression/

from jax import config

config.update("jax_enable_x64", True)

import gpjax as gpx
import jax
from jax import jit
import optax as ox
import jax.random as jr

from gpjax.objectives import ConjugateMLL
from gpjax.mean_functions import Zero
from evox.operators.gaussian_process.kernels import RBF




class GPRegression:
    """
    Gaussian Process Regreesion model using JAX, Optax and GPJAX.

    This model applies Gaussian Process (GP) for regression tasks. It uses an RBF kernel and
    a Zero mean function by default. The class allows for fitting the model to data and making predictions.
    The GP's prior and posterior are defined, and the model is fitted using the specified optimization method.

    Attributes:
        kernel: The kernel function for the GP (default is RBF).
                The difference between the kernels are shown in the link: https://docs.jaxgaussianprocesses.com/examples/intro_to_kernels/.
        mean_fun: The mean function for the GP (default is Zero).
        key: JAX random key for stochastic operations.
        prior: The prior distribution of the GP, which is chosen by the user according to the specific problems.
        likelihood: The likelihood function for the GP.
        object: The objective function (default is Conjugate Marginal Log Likelihood).
        posterior: The posterior distribution of the GP after applying the likelihood.
    """

    def __init__(
        self,
        kernel=RBF(),
        meanfun=Zero(),
        likelihood=None,
        object=ConjugateMLL(negative=True),
        key=jr.PRNGKey(123),
    ):
        """Initializes the Gaussian Process Regression model with default parameters."""
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
        """
        Fits the model to the provided data.

        The method takes in features (x) and labels (y), creates a dataset, and then uses
        the specified optimization method to fit the model. The process includes setting
        the objective function and optimizing the posterior distribution.

        Args:
            x: The feature matrix.
            y: The label vector.
            optimizer: The optimization algorithm implemented by Optax to use (default is Gradient Transformation).
        """
        self.opt_posterior, self.history = gpx.fit(
            model=self.posterior,
            objective=self.object,
            train_data=self.dataset,
            optim=optimzer,
            num_iters=500,
            key=self.key,
        )

    def fit_scipy(self, x, y):
        self.dataset = gpx.Dataset(X=x, y=y)
        self.object(self.posterior, train_data=self.dataset)
        self.opt_posterior, self.history = gpx.fit_scipy(
            model=self.posterior,
            objective=self.object,
            train_data=self.dataset,
        )

    def predict(self, x):
        """
        Makes predictions on new data.

        The method computes the predictive distribution of the optimized model for new input features.
        It returns the predictive distribution along with the mean and standard deviation of the predictions.

        Args:
            x: The feature matrix for which predictions are to be made.

        Returns:
            A tuple containing the predictive distribution, predictive mean, and predictive standard deviation.
        """        
        latent_dist = self.opt_posterior.predict(x, train_data=self.dataset)
        predictive_dist = self.opt_posterior.likelihood(latent_dist)

        predictive_mean = predictive_dist.mean()
        predictive_std = predictive_dist.stddev()
        return predictive_dist, predictive_mean, predictive_std

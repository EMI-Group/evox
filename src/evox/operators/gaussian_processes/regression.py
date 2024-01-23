# https://docs.jaxgaussianprocesses.com/examples/regression/

from jax import config
from jax import jit
import jax.numpy as jnp
import jax.random as jr
import gpjax as gpx
from gpjax.kernels import RBF
from gpjax.mean_functions import Zero
from gpjax.likelihoods import Gaussian
from gpjax.objectives import ConjugateMLL

# the demand of gpJAX. Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)
class GPRegression:
    def __init__(self, kernel=RBF(), meanfun=Zero(), likelihood=Gaussian(), optimizer=ConjugateMLL(negative=True),key=jr.PRNGKey(123)):
        self.kernel = kernel
        self.mean_fun = meanfun
        self.key = key
        self.prior = gpx.gps.Prior(mean_function=meanfun, kernel=kernel)
        self.likelihood = likelihood
        self.optimizer = optimizer





    def fit(self, x, y):
        self.dataset = gpx.Dataset(X=x, y=y)
        self.likelihood.num_datapoints = self.dataset.n
        self.posterior = self.prior * self.likelihood
        self.optimizer(self.posterior, train_data=self.dataset)
        # add jit
        self.opt_posterior, self.history = gpx.fit_scipy(
            model= self.posterior,
            objective= self.optimizer,
            train_data= self.dataset,
        )



    def predict(self,x):
        latent_dist = self.opt_posterior.predict(x, train_data=self.dataset)
        predictive_dist = self.opt_posterior.likelihood(latent_dist)

        predictive_mean = predictive_dist.mean()
        predictive_std = predictive_dist.stddev()
        return predictive_mean, predictive_std

    # def sample_y(self, x, num_sample=1, random_state = 0):

#
#
# key = jr.PRNGKey(123)
# plt.style.use(
#     "https://raw.githubusercontent.com/JaxGaussianProcesses/GPJax/main/docs/examples/gpjax.mplstyle"
# )
# cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
#
# n = 100
# noise = 0.3
#
# key, subkey = jr.split(key)
# x = jr.uniform(key=key, minval=-3.0, maxval=3.0, shape=(n,)).reshape(-1, 1)
# f = lambda x: jnp.sin(4 * x) + jnp.cos(2 * x)
# signal = f(x)
# y = signal + jr.normal(subkey, shape=signal.shape) * noise
#
# D = gpx.Dataset(X=x, y=y)
#
# xtest = jnp.linspace(-3.5, 3.5, 500).reshape(-1, 1)
# ytest = f(xtest)
#
# fig, ax = plt.subplots()
# ax.plot(x, y, "o", label="Observations", color=cols[0])
# ax.plot(xtest, ytest, label="Latent function", color=cols[1])
# ax.legend(loc="best")
#
# kernel = gpx.kernels.RBF()
# meanf = gpx.mean_functions.Zero()
# prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
#
# prior_dist = prior.predict(xtest)
#
# prior_mean = prior_dist.mean()
# prior_std = prior_dist.variance()
# samples = prior_dist.sample(seed=key, sample_shape=(20,))
#
#
# fig, ax = plt.subplots()
# ax.plot(xtest, samples.T, alpha=0.5, color=cols[0], label="Prior samples")
# ax.plot(xtest, prior_mean, color=cols[1], label="Prior mean")
# ax.fill_between(
#     xtest.flatten(),
#     prior_mean - prior_std,
#     prior_mean + prior_std,
#     alpha=0.3,
#     color=cols[1],
#     label="Prior variance",
# )
# ax.legend(loc="best")
# clean_legend(ax)
#
# likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)
#
# posterior = prior * likelihood
#
# negative_mll = gpx.objectives.ConjugateMLL(negative=True)
# negative_mll(posterior, train_data=D)
#
#
# # static_tree = jax.tree_map(lambda x: not(x), posterior.trainables)
# # optim = ox.chain(
# #     ox.adam(learning_rate=0.01),
# #     ox.masked(ox.set_to_zero(), static_tree)
# #     )
#
# negative_mll = jit(negative_mll)
#
# opt_posterior, history = gpx.fit_scipy(
#     model=posterior,
#     objective=negative_mll,
#     train_data=D,
# )
#
# latent_dist = opt_posterior.predict(xtest, train_data=D)
# predictive_dist = opt_posterior.likelihood(latent_dist)
#
# predictive_mean = predictive_dist.mean()
# predictive_std = predictive_dist.stddev()

from evox.operators.gaussian_processes import regression
import jax
import jax.numpy as jnp
import jax.random as jr
from gpjax.likelihoods import Gaussian
import gpjax as gpx
from evox.problems.numerical.maf import MaF3
import optax as ox

# from scipy.optimizer.
key = jr.PRNGKey(123)
n = 100
noise = 0.3
d = 12
m =3

key, subkey = jr.split(key)
x = jr.uniform(key=key, minval=-3.0, maxval=3.0, shape=(n,d))

prob = MaF3(d=d, m=m)
state = prob.init(key)
y, new_state1 = prob.evaluate(state, x)
y_ = y[:,0:1]
print(y_.shape)
D = gpx.Dataset(X=x, y=y_)
print(D.n)
linspace_points = jnp.linspace(0, 1, 500)
# xtest = jnp.meshgrid(linspace_points, linspace_points, linspace_points, linspace_points,
#                       linspace_points, linspace_points, linspace_points, linspace_points,
#                       linspace_points, linspace_points, linspace_points, linspace_points)[0

xtest = jax.random.uniform(key, (500, 12), minval=-3.5, maxval=3.5)

# print(D.n)
likelihood = Gaussian(num_datapoints=D.n)
gp = regression.GPRegression(likelihood=likelihood)
optimizer = ox.adagrad(learning_rate=1e-3)
gp.fit(x=x, y=y_, optimzer=optimizer)
predictive_mean, predictive_std = gp.predict(xtest)

from jax import jit, random, vmap
import jax.numpy as jnp
from evox import jit_class
from functools import partial


@partial(jit, static_argnames=["num_sample"])
def bounding_cube_monte_carlo_hv(key, objs, ref, num_sample):
    # move ref to (0, 0, 0, ...), and take the absolute value
    points = jnp.abs(objs - ref)
    _num_points, num_objs = points.shape
    bound = jnp.max(points, axis=0)
    # calculate the max volume possible (non-overlapping)
    max_vol = jnp.prod(bound)
    samples = random.uniform(key, shape=(num_sample, num_objs), minval=0, maxval=bound)
    # samples are valid if they're inside any one of the hypercube.
    in_hypercube = jnp.sum(
        vmap(
            lambda sample, points: jnp.any(jnp.all(sample < points, axis=1)),
            in_axes=(0, None),
        )(samples, points)
    )
    return in_hypercube / num_sample * max_vol


def dominate_count(sample, points):
    # sample (m, )
    # points (n, m)
    return jnp.sum(jnp.all(sample < points, axis=1))


@partial(jit, static_argnames=["num_sample"])
def calc_contribution(key, points, current_point, num_sample):
    """calculate the contribution of each hypercube
    overlapping areas contribute 1/n where n is the number of overlaps
    """
    num_points, num_objs = points.shape
    samples = random.uniform(
        key, shape=(num_sample, num_objs), minval=0, maxval=current_point
    )
    dom_count = vmap(dominate_count, in_axes=(0, None))(samples, points)
    bins = jnp.bincount(dom_count, length=num_points + 1)
    # calculate the max volume possible (non-overlapping)
    max_vol = jnp.prod(current_point)
    return max_vol * jnp.sum(bins[1:] / (jnp.arange(num_points) + 1)) / num_sample


@partial(jit, static_argnames=["num_sample"])
def each_cube_monte_carlo_hv(key, objs, ref, num_sample):
    # move ref to (0, 0, 0, ...), and take the absolute value
    points = jnp.abs(objs - ref)
    num_points = points.shape[0]
    keys = random.split(key, num_points)
    # calculate the contribution of each hypercube and sum them
    return jnp.sum(
        vmap(calc_contribution, in_axes=(0, None, 0, None))(
            keys, points, points, num_sample // num_points
        )
    )


@jit_class
class HV:
    """Hypervolume indicator
    Implemented using monte carlo.
    We offers two different sample methods: `bounding_cube` and `each_cube`.
    With `bounding_cube`, we draw samples from a hypercube that can bound all objectives.
    With `each_cube`, we draw samples from each hypercube form by each objective and the reference point.
    Since reference point is often far from pf, `bounding_cube` method usually gives more accurate result.
    """

    def __init__(self, ref, num_sample=100_000, sample_method="bounding_cube"):
        """
        Parameters
        ----------
        ref
            The reference point.
        num_sample
            Number of samples to draw when doing monte carlo.
        sample_method
            `bounding_cube` or `each_cube`.
            Default to `bounding_cube`.
        """
        self.ref = ref
        self.num_sample = num_sample
        if sample_method == "bounding_cube":
            self.hv_impl = bounding_cube_monte_carlo_hv
        elif sample_method == "each_cube":
            self.hv_impl = each_cube_monte_carlo_hv
        else:
            raise ValueError(
                f"sample_method should be 'bounding_cube' or 'each_cube', got '{sample_method}'."
            )

    def __call__(self, key, objs):
        return self.hv_impl(key, objs, self.ref, self.num_sample)

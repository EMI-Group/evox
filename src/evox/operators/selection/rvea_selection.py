import jax
import jax.numpy as jnp
from jax import jit
from evox.utils import cos_dist
from evox import jit_class


@jit
def ref_vec_guided(x, f, v, theta):
    n, m = jnp.shape(f)
    nv = jnp.shape(v)[0]

    obj = f - jnp.nanmin(f, axis=0)
    obj = jnp.maximum(obj, 1e-32)

    cosine = cos_dist(v, v)
    cosine = jnp.where(jnp.eye(jnp.shape(cosine)[0], dtype=bool), 0, cosine)
    cosine = jnp.clip(cosine, 0, 1)
    gamma = jnp.min(jnp.arccos(cosine), axis=1)

    angle = jnp.arccos(jnp.clip(cos_dist(obj, v), 0, 1))

    nan_mask = jnp.isnan(obj).any(axis=1)
    associate = jnp.argmin(angle, axis=1)
    associate = jnp.where(nan_mask, -1, associate)

    partition = jax.vmap(
        lambda x: jnp.where(associate == x, jnp.arange(0, n), -1), in_axes=1, out_axes=1
    )(jnp.tile(jnp.arange(0, nv), (n, 1)))

    mask = partition == -1
    mask_null = jnp.sum(mask, axis=0) == n

    apd = jax.vmap(
        lambda x, y, z: (1 + m * theta * z[x] / y)
        * jnp.sqrt(jnp.sum(obj[x, :] ** 2, axis=1)),
        in_axes=(1, 0, 1),
        out_axes=1,
    )(partition, gamma, angle)
    apd = jnp.where(mask, jnp.inf, apd)

    next_ind = jnp.argmin(apd, axis=0)
    next_x = jnp.where(mask_null[:, jnp.newaxis], jnp.nan, x[next_ind])
    next_f = jnp.where(mask_null[:, jnp.newaxis], jnp.nan, f[next_ind])

    return next_x, next_f


@jit_class
class ReferenceVectorGuided:
    """Reference vector guided environmental selection."""

    def __call__(self, x, f, v, theta):
        return ref_vec_guided(x, f, v, theta)

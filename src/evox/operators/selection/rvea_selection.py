import jax.numpy as jnp
from jax import jit, lax
from evox.utils import cos_dist
from evox import jit_class


@jit
def ref_vec_guided(x, v, theta):
    n, m = jnp.shape(x)
    nv = jnp.shape(v)[0]
    obj = x

    obj -= jnp.tile(jnp.min(obj, axis=0), (n, 1))

    cosine = cos_dist(v, v)
    cosine = jnp.where(jnp.eye(jnp.shape(cosine)[0], dtype=bool), 0, cosine)
    cosine = jnp.clip(cosine, -1, 1)
    gamma = jnp.min(jnp.arccos(cosine), axis=1)

    angle = jnp.arccos(cos_dist(obj, v))

    associate = jnp.argmin(angle, axis=1)

    next_ind = jnp.full(nv, -1)
    is_null = jnp.sum(next_ind)
    global_min = jnp.inf
    global_min_idx = -1

    vals = next_ind, global_min, global_min_idx

    def update_next(i, sub_index, next_ind, global_min, global_min_idx):
        apd = (1 + m * theta * angle[sub_index, i] / gamma[i]) * jnp.sqrt(
            jnp.sum(obj[sub_index, :] ** 2, axis=1)
        )

        apd_max = jnp.max(apd)
        noise = jnp.where(sub_index == -1, apd_max, 0)
        apd = apd + noise
        best = jnp.argmin(apd)

        global_min_idx = jnp.where(
            apd[best] < global_min, sub_index[best.astype(int)], global_min_idx
        )
        global_min = jnp.minimum(apd[best], global_min)

        next_ind = next_ind.at[i].set(sub_index[best.astype(int)])
        return next_ind, global_min, global_min_idx

    def no_update(i, sub_index, next_ind, global_min, global_min_idx):
        return next_ind, global_min, global_min_idx

    def body_fun(i, vals):
        next_ind, global_min, global_min_idx = vals
        sub_index = jnp.where(associate == i, size=nv, fill_value=-1)[0]

        next_ind, global_min, global_min_idx = lax.cond(
            jnp.sum(sub_index) != is_null,
            update_next,
            no_update,
            i,
            sub_index,
            next_ind,
            global_min,
            global_min_idx,
        )
        return next_ind, global_min, global_min_idx

    next_ind, global_min, global_min_idx = lax.fori_loop(0, nv, body_fun, vals)
    mask = next_ind == -1

    next_ind = jnp.where(mask, global_min_idx, next_ind)
    next_ind = jnp.where(global_min_idx != -1, next_ind, jnp.arange(0, nv))

    return next_ind


@jit_class
class ReferenceVectorGuided:
    """Reference vector guided environmental selection."""

    def __call__(self, x, v, theta):
        return ref_vec_guided(x, v, theta)

import jax
import jax.numpy as jnp
from evox import jit_class
from evox.utils import pairwise_euclidean_dist
from evox.operators.non_dominated_sort import _dominate_relation


@jit_class
class KnEASelection:
    """Mating selection (in KnEA algorithm)
    
    A binary tournament selection based on knee points and neighbors
    """

    def __init__(self, num_round: int, k_neighbors: int = 3):
        self.num_round = num_round
        self.k = k_neighbors

    def __call__(self, key, pop, fit, knee):
        relation = _dominate_relation(fit, fit)
        dis = pairwise_euclidean_dist(fit, fit)
        order = jnp.argsort(dis, axis=1)
        neighbor = jnp.take_along_axis(dis, order[:, 1 : self.k + 1], axis=1)
        avg = jnp.sum(neighbor, axis=1) / self.k
        r = 1 / abs(neighbor - avg[:, None])
        w = r / jnp.sum(r, axis=1)[:, None]
        DW = jnp.sum(neighbor * w, axis=1)
        chosen = jax.random.choice(key, self.num_round, shape=(self.num_round, 2))
        a, b = chosen[:, 0], chosen[:, 1]
        winners = (relation[b, a] |
                   (~relation[a, b] & ((knee[b] & ~knee[a]) |
                    (~(knee[a] & ~knee[b]) & (DW[b] > DW[a]))))) * 1
        idx = chosen[:, winners].diagonal()
        return pop[idx]
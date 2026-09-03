import etl
import etl.random as random


def polynomial_mutation(key, x, lb, ub, pro_m: float = 1.0, dis_m: float = 20.0):
    """Polynomial mutation.
    Inspired by PlatEMO.

    :param key: The random key.
    :param x: The input population (size: n x d).
    :param lb: The lower bounds for the decision variables.
    :param ub: The upper bounds for the decision variables.
    :param pro_m: Probability of mutation.
    :param dis_m: The distribution index for polynomial mutation.

    :return: The mutated population. (size: n x d)
    """
    n, d = x.shape[0], x.shape[1]
    k1, k2 = random.split(key)
    # Random numbers for mutation
    site = random.uniform(k1, (n, d), dtype="float32") < pro_m / d
    mu = random.uniform(k2, (n, d), dtype="float32")
    # Apply mutation for the first part where mu <= 0.5
    temp = etl.logical_and(site, mu <= 0.5)
    lower = lb
    upper = ub

    pop_dec = etl.maximum(etl.minimum(x, upper), lower)

    norm = etl.select(temp, (pop_dec - lower) / (upper - lower), 0.0)

    pop_dec = etl.select(
        temp,
        pop_dec
        + (upper - lower)
        * (
            etl.power(
                2 * mu + (1 - 2 * mu) * etl.power(1 - norm, dis_m + 1.0),
                1.0 / (dis_m + 1.0),
            )
            - 1.0
        ),
        pop_dec,
    )

    # Apply mutation for the second part where mu > 0.5
    temp = etl.logical_and(site, mu > 0.5)
    norm = etl.select(temp, (upper - pop_dec) / (upper - lower), 0.0)
    pop_dec = etl.select(
        temp,
        pop_dec
        + (upper - lower)
        * (
            1.0
            - etl.power(
                2 * (1 - mu) + 2 * (mu - 0.5) * etl.power(1 - norm, dis_m + 1.0),
                1.0 / (dis_m + 1.0),
            )
        ),
        pop_dec,
    )

    return pop_dec

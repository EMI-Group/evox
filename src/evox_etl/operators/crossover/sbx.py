import etl
import etl.random as random


def simulated_binary(key, x, pro_c: float = 1.0, dis_c: float = 20.0):
    """Simulated binary crossover (SBX)

    :param key: The random key.
    :param x: Parent solutions in a 2D tensor (size: n x d).
    :param pro_c: Probability of doing crossover.
    :param dis_c: Distribution index of SBX.

    :return: Offspring solutions after crossover.
    """
    n, m = x.shape
    n2 = n // 2
    parent1_dec = x[:n2]
    parent2_dec = x[n2 : n2 * 2]

    k1, k2, k3, k4 = random.split_n(key, 4)

    # Uniform distribution for mutation
    mu = random.uniform(k1, (n2, m), dtype="float32")

    # Beta calculation for SBX
    beta = etl.select(mu <= 0.5, etl.power(2 * mu, 1.0 / (dis_c + 1.0)), 0.0)
    beta = etl.select(mu > 0.5, etl.power(2.0 - 2 * mu, -1.0 / (dis_c + 1.0)), beta)

    # Random binary for mutation direction
    beta = beta * (1 - random.randint(k2, (n2, m), 0, 2, "int32") * 2)

    # Apply crossover probability to mutate
    beta = etl.select(random.uniform(k3, (n2, m), dtype="float32") < 0.5, 1.0, beta)
    beta = etl.select(random.uniform(k4, (n2, m), dtype="float32") > pro_c, 1.0, beta)

    mid = (parent1_dec + parent2_dec) / 2
    half_diff = (parent1_dec - parent2_dec) / 2
    offspring_dec = etl.concatenate([mid + beta * half_diff, mid - beta * half_diff], axis=0)

    return offspring_dec

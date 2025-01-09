import torch


def simulated_binary(x: torch.Tensor, pro_c: float = 1, dis_c: float = 20):
    """
    Simulated binary crossover (SBX)
    Args:
        x: Parent solutions in a 2D tensor (size: n x d).
        pro_c: Probability of doing crossover.
        dis_c: Distribution index of SBX.
    Returns:
        offspring_dec: Offspring solutions after crossover.
    """
    n, _ = x.size()
    parent1_dec = x[: n // 2, :]
    parent2_dec = x[n // 2 : n // 2 * 2, :]

    # Uniform distribution for mutation
    mu = torch.rand_like(parent1_dec)

    # Beta calculation for SBX
    beta = torch.zeros_like(parent1_dec)
    beta = torch.where(mu <= 0.5, torch.pow(2 * mu, 1 / (dis_c + 1)), beta)
    beta = torch.where(mu > 0.5, torch.pow(2 - 2 * mu, -1 / (dis_c + 1)), beta)

    # Random binary for mutation direction
    beta = beta * (1 - torch.randint(0, 2, beta.size(), device=x.device) * 2)

    # Apply crossover probability to mutate
    beta = torch.where(torch.rand_like(beta) < 0.5, 1, beta)
    beta = torch.where(torch.rand_like(beta) > pro_c, 1, beta)

    offspring_dec = torch.cat(
        [
            (parent1_dec + parent2_dec) / 2 + beta * (parent1_dec - parent2_dec) / 2,
            (parent1_dec + parent2_dec) / 2 - beta * (parent1_dec - parent2_dec) / 2,
        ]
    )

    return offspring_dec

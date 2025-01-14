import torch


def simulated_binary(x: torch.Tensor, pro_c: float = 1.0, dis_c: float = 20.0) -> torch.Tensor:
    """Simulated binary crossover (SBX)

    :param x: Parent solutions in a 2D tensor (size: n x d).
    :param pro_c: Probability of doing crossover.
    :param dis_c: Distribution index of SBX.

    :return: Offspring solutions after crossover.
    """
    n, m = x.size()
    parent1_dec = x[: n // 2, :]
    parent2_dec = x[n // 2 : n // 2 * 2, :]

    # Uniform distribution for mutation
    mu = torch.rand(n // 2, m, device=x.device)

    # Beta calculation for SBX
    beta = torch.zeros(mu.size(), device=x.device)
    beta = torch.where(mu <= 0.5, torch.pow(2 * mu, 1 / (dis_c + 1)), beta)
    beta = torch.where(mu > 0.5, torch.pow(2 - 2 * mu, -1 / (dis_c + 1)), beta)

    # Random binary for mutation direction
    beta = beta * (1 - torch.randint(0, 2, beta.size(), device=x.device) * 2)

    # Apply crossover probability to mutate
    beta = torch.where(torch.rand(mu.size(), device=x.device) < 0.5, 1, beta)
    beta = torch.where(torch.rand(mu.size(), device=x.device) > pro_c, 1, beta)

    offspring_dec = torch.cat(
        [
            (parent1_dec + parent2_dec) / 2 + beta * (parent1_dec - parent2_dec) / 2,
            (parent1_dec + parent2_dec) / 2 - beta * (parent1_dec - parent2_dec) / 2,
        ]
    )

    return offspring_dec

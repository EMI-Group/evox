import torch

from evox.utils import maximum, minimum


def polynomial_mutation(
    x: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
    pro_m: float = 1,
    dis_m: float = 20,
) -> torch.Tensor:
    """Polynomial mutation.
    Inspired by PlatEMO.

    :param x: The input population (size: n x d).
    :param lb: The lower bounds for the decision variables.
    :param ub: The upper bounds for the decision variables.
    :param pro_m: Probability of mutation.
    :param dis_m: The distribution index for polynomial mutation.

    :return: The mutated population. (size: n x d)
    """
    n, d = x.size()
    # Random numbers for mutation
    site = torch.rand(n, d, device=x.device) < pro_m / d
    mu = torch.rand(n, d, device=x.device)
    # Apply mutation for the first part where mu <= 0.5
    temp = site & (mu <= 0.5)
    lower = lb
    upper = ub

    pop_dec = maximum(minimum(x, upper), lower)

    norm = torch.where(temp, (pop_dec - lower) / (upper - lower), 0.0)

    pop_dec = torch.where(
        temp,
        pop_dec
        + (upper - lower)
        * (
            torch.pow(
                2 * mu + (1 - 2 * mu) * torch.pow(1 - norm, dis_m + 1),
                1 / (dis_m + 1),
            )
            - 1
        ),
        pop_dec,
    )

    # Apply mutation for the second part where mu > 0.5
    temp = site & (mu > 0.5)
    norm = torch.where(temp, (upper - pop_dec) / (upper - lower), torch.zeros_like(pop_dec))
    pop_dec = torch.where(
        temp,
        pop_dec
        + (upper - lower)
        * (
            1
            - torch.pow(
                2 * (1 - mu) + 2 * (mu - 0.5) * torch.pow(1 - norm, dis_m + 1),
                1 / (dis_m + 1),
            )
        ),
        pop_dec,
    )

    return pop_dec

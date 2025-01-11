import torch

from ...utils import clamp


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
    :param lb: The lb boundary for the mutation.
    :param ub: The ub boundary for the mutation.
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
    pop_dec = clamp(x, lb, ub)
    norm = torch.where(temp, (pop_dec - lb) / (ub - lb), 0.0)
    pop_dec = torch.where(
        temp,
        pop_dec
        + (ub - lb)
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
    norm = torch.where(temp, (ub - pop_dec) / (ub - lb), torch.zeros_like(pop_dec))
    pop_dec = torch.where(
        temp,
        pop_dec
        + (ub - lb)
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

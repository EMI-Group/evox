import torch
import numpy as np
from ..utils import maximum, minimum


def polynomial_mutation(x: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor, pro_m: float = 1, dis_m: float = 20):
    """Polynomial mutation
    Inspired by PlatEMO.

    Args:
        boundary: The lower and upper boundary for the mutation.
        pro_m: Probability of mutation.
        dis_m: The distribution index for polynomial mutation.
    """
    # if x.shape[0] % 2 != 0:
    #     pop_dec = x[:(x.shape[0] // 2) * 2, :]
    n, d = x.shape
    # Random numbers for mutation
    site = (torch.rand(n, d) < pro_m / d).to(x.device)
    mu = torch.rand(n, d).to(x.device)
    # Apply mutation for the first part where mu <= 0.5
    temp = (site & (mu <= torch.tensor(0.5, device=x.device)))
    lower = lb
    upper = ub
    # lower = torch.tile(torch.tensor(lb, dtype=torch.float32), (n, 1))
    # upper = torch.tile(torch.tensor(ub, dtype=torch.float32), (n, 1))
    pop_dec = maximum(minimum(x, upper), lower)

    norm = torch.where(temp, (pop_dec - lower) / (upper - lower), torch.tensor(0.0, device=x.device))

    pop_dec = torch.where(
        temp,
        pop_dec
        + (upper - lower)
        * (
                torch.pow(
                    torch.tensor(2.0, device=x.device) * mu + (torch.tensor(1.0, device=x.device) - torch.tensor(2.0,
                                                                                                                 device=x.device) * mu) * torch.pow(
                        torch.tensor(1.0, device=x.device) - norm,
                        torch.tensor(dis_m, device=x.device) + torch.tensor(1.0, device=x.device)),
                    torch.tensor(1.0, device=x.device) / (
                                torch.tensor(dis_m, device=x.device) + torch.tensor(1.0, device=x.device)),
                )
            - torch.tensor(1.0, device=x.device)
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
            torch.tensor(1.0, device=x.device)
            - torch.pow(
                torch.tensor(2.0, device=x.device) * (torch.tensor(1.0, device=x.device) - mu)
                + torch.tensor(2.0, device=x.device) * (mu - torch.tensor(0.5, device=x.device)) * torch.pow(torch.tensor(1.0, device=x.device) - norm, dis_m + torch.tensor(1.0, device=x.device)),
                torch.tensor(1.0, device=x.device) / (dis_m + torch.tensor(1.0, device=x.device)),
            )
        ),
        pop_dec,
    )

    # If the number of individuals was odd, append the last one back
    # if x.shape[0] % 2 != 0:
    #     pop_dec = torch.cat([pop_dec, x[-1:, :]], dim=0)


        # Apply mutation for the second part where mu > 0.5
        temp = site & (mu > 0.5)
        norm = torch.where(temp, (upper - pop_dec) / (upper - lower), torch.zeros_like(pop_dec))
        pop_dec = torch.where(
            temp,
            pop_dec
            + (upper - lower)
            * (
                1.0
                - torch.pow(
                    2.0 * (1.0 - mu)
                    + 2.0 * (mu - 0.5) * torch.pow(1.0 - norm, dis_m + 1.0),
                    1.0 / (dis_m + 1.0),
                )
            ),
            pop_dec,
        )

        # If the number of individuals was odd, append the last one back
        if n % 2 != 0:
            pop_dec = torch.cat([pop_dec, x[-1:, :]], dim=0)

        return pop_dec

    def __call__(self, x):
        return self._polynomial(x, self.boundary, self.pro_m, self.dis_m)


if __name__ == "__main__":
    n_individuals = 3
    n_genes = 10
    x = torch.randn(n_individuals, n_genes)
    print(x)

    boundary = np.array([[-1] * n_genes, [1] * n_genes])

    polynomial = Polynomial(boundary=boundary, pro_m=1, dis_m=20)

    offspring = polynomial(x)

    print("Offspring Shape:", offspring.shape)
    print(offspring)

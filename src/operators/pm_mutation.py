import torch
import numpy as np

class Polynomial:
    """Polynomial mutation
    Inspired by PlatEMO.

    Args:
        boundary: The lower and upper boundary for the mutation.
        pro_m: Probability of mutation.
        dis_m: The distribution index for polynomial mutation.
    """

    def __init__(self, boundary, pro_m=1, dis_m=20):
        self.boundary = boundary
        self.pro_m = pro_m
        self.dis_m = dis_m

    def _polynomial(self, x, boundary, pro_m, dis_m):
        if x.shape[0] % 2 != 0:
            x = x[:(x.shape[0] // 2) * 2, :] # Ensure n is even
        n, d = x.shape
        # Random numbers for mutation
        site = torch.rand(n, d) < pro_m / d
        mu = torch.rand(n, d)

        # Apply mutation for the first part where mu <= 0.5
        temp = site & (mu <= 0.5)
        lower, upper = torch.tile(torch.tensor(boundary[0]), (n, 1)), torch.tile(torch.tensor(boundary[1]), (n, 1))
        pop_dec = torch.maximum(torch.minimum(x, upper), lower)  # Apply boundary constraints
        norm = torch.where(temp, (pop_dec - lower) / (upper - lower), torch.zeros_like(pop_dec))

        pop_dec = torch.where(
            temp,
            pop_dec
            + (upper - lower)
            * (
                torch.pow(
                    2.0 * mu + (1.0 - 2.0 * mu) * torch.pow(1.0 - norm, dis_m + 1.0),
                    1.0 / (dis_m + 1),
                )
                - 1.0
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

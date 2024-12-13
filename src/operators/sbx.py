import torch

class SimulatedBinary:
    """Simulated binary crossover (SBX)
    Args:
        pro_c: Probability of doing crossover.
        dis_c: Distribution index of SBX.
        type: Type of crossover. Type 1 generates offspring of the same size as the parent.
              Type 2 generates offspring of half the population size of the parent.
    """

    def __init__(self, pro_c=1, dis_c=20, type=1):
        self.pro_c = pro_c
        self.dis_c = dis_c
        self.type = type

    def _simulated_binary(self, x, pro_c, dis_c, type):
        n, _ = x.shape
        parent1_dec = x[:n // 2, :]
        parent2_dec = x[n // 2:n // 2 * 2, :]
        n_p, d = parent1_dec.shape

        # Uniform distribution for mutation
        mu = torch.rand_like(parent1_dec)

        # Beta calculation for SBX
        beta = torch.zeros_like(parent1_dec)
        beta = torch.where(mu <= 0.5, torch.pow(2 * mu, 1 / (dis_c + 1)), beta)
        beta = torch.where(mu > 0.5, torch.pow(2 - 2 * mu, -1 / (dis_c + 1)), beta)

        # Random binary for mutation direction
        # Replace randint_like with torch.randint
        beta *= (-1) ** torch.randint(0, 2, beta.shape, dtype=torch.int32)

        # Apply crossover probability to mutate
        beta = torch.where(torch.rand_like(beta) < 0.5, 1, beta)
        beta = torch.where(torch.rand_like(beta) > pro_c, 1, beta)

        # Type 1: Generate offspring of the same size as the parent
        if type == 1:
            offspring_dec = torch.cat(
                [
                    (parent1_dec + parent2_dec) / 2 + beta * (parent1_dec - parent2_dec) / 2,
                    (parent1_dec + parent2_dec) / 2 - beta * (parent1_dec - parent2_dec) / 2,
                ]
            )
            if n % 2 != 0:
                offspring_dec = torch.cat([offspring_dec, x[-1:, :]], dim=0)

        # Type 2: Generate offspring of half the population size
        elif type == 2:
            offspring_dec = (parent1_dec + parent2_dec) / 2 + beta * (parent1_dec - parent2_dec) / 2

        return offspring_dec

    def __call__(self, x):
        return self._simulated_binary(x, self.pro_c, self.dis_c, self.type)


if __name__ == "__main__":
    n_individuals = 4
    n_genes = 10
    x = torch.randn(n_individuals, n_genes)
    print(x)

    sbx = SimulatedBinary(pro_c=1, dis_c=20, type=2)

    offspring = sbx(x)

    print("Offspring Shape:", offspring.shape)
    print(offspring)

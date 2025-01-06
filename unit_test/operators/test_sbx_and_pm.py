import torch
from src.operators import polynomial_mutation, simulated_binary


if __name__ == "__main__":
    n_individuals = 3
    n_genes = 10
    torch.manual_seed(42)
    x = torch.randn(n_individuals, n_genes)

    offspring = simulated_binary(x)
    print(offspring)

    # Call polynomial mutation function
    offspring = polynomial_mutation(
        x,
        lb=torch.tensor([[-1] * n_genes]),
        ub=torch.tensor([[1] * n_genes]),
        pro_m=1,
        dis_m=20,
    )

    print(offspring)

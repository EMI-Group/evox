import torch
from src.operators import ref_vec_guided

if __name__ == "__main__":
    # Generate random test data for x, f, v
    seed = 42
    torch.manual_seed(seed)

    n, m, nv = 12, 4, 5
    x = torch.randn(n, 10)  # Random solutions
    print(x)
    f = torch.randn(n, m)   # Random objective values
    f[1] = torch.tensor([float('nan')] * m)
    print(f)
    v = torch.randn(nv, m)  # Random reference vectors
    print(v)
    theta = torch.tensor(0.5)  # Arbitrary theta value

    next_x, next_f = ref_vec_guided(x, f, v, theta)

    print("Next x:", next_x)
    print("Next f:", next_f)
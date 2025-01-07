import torch
from src.operators.selection import ref_vec_guided
from src.core import jit, vmap


if __name__ == "__main__":
    # Generate random test data for x, f, v
    seed = 42
    torch.manual_seed(seed)

    n, m, nv = 12, 4, 5
    x = torch.randn(n, 10)  # Random solutions
    f = torch.randn(n, m)  # Random objective values
    f[1] = torch.tensor([float("nan")] * m)

    v = torch.randn(nv, m)
    theta = torch.tensor(0.5)  # Arbitrary theta value

    jit_ref_vec_guided = jit(ref_vec_guided, trace=True, lazy=True)
    next_x, next_f = ref_vec_guided(x, f, v, theta)
    next_x1, next_f1 = jit_ref_vec_guided(x, f, v, theta)

    print("Next x:", next_x)
    print("Next f:", next_f)

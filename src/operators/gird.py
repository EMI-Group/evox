import torch


class GridSampling:
    """
    Grid sampling.
    Inspired by PlatEMO.
    """

    def __init__(self, n=None, m=None):
        self.n = n
        self.m = m
        self.num_points = int(torch.ceil(torch.tensor(self.n ** (1 / self.m))).item())

    def __call__(self):
        gap = torch.linspace(0, 1, self.num_points)
        grid_axes = [gap for _ in range(self.m)]

        # Generate grid using meshgrid and stack values
        grid_values = torch.meshgrid(*grid_axes, indexing="ij")

        # Stack grids along the last axis (axis=-1)
        w = torch.stack(grid_values, dim=-1).reshape(-1, self.m)

        # Reverse the order of columns to match JAX's `w[:, ::-1]`
        w = w.flip(dims=[1])

        n = w.shape[0]
        return w, n


if __name__ == "__main__":
    n = 9
    m = 2
    grid_sampler = GridSampling(n, m)

    w, num_samples = grid_sampler()

    print("Generated grid points (w):")
    print(w)
    print("\nNumber of generated samples:", num_samples)

    print("\nShape of w:", w.shape)
    assert w.shape == (num_samples, m), f"Expected shape {(num_samples, m)}, but got {w.shape}"

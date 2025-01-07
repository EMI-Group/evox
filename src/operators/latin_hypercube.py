import torch


class LatinHypercubeSampling:
    def __init__(
        self,
        d: int,
        scale: torch.Tensor = None,
        scramble: bool = True,
        rng_seed: int = None
    ):
        """
        Initialize the Latin Hypercube Sampling (LHS) generator.

        Parameters:
        ----------
        d: int
            The dimensionality of the sample space.
        scale: torch.Tensor of shape (d, 2), optional
            The range of the sample space in each dimension. If None, defaults to [0, 1] for all dimensions.
        scramble: bool, default=True
            Whether to scramble the sample points. Scrambling randomizes the order of points in each dimension.
        rng_seed: int, default=None
            The seed for the random number generator (optional).
        """
        self.d = d
        self.scale = scale if scale is not None else torch.tensor([[0.0] * d, [1.0] * d])
        self.scramble = scramble
        self.rng = torch.manual_seed(rng_seed) if rng_seed is not None else torch.random

    def sample(self, n: int) -> torch.Tensor:
        """
        Generate Latin Hypercube samples.

        Parameters:
        ----------
        n: int
            The number of sample points to generate.

        Returns:
        -------
        torch.Tensor
            A tensor of shape (n, d), where each row represents a sample point and each column represents a dimension.
        """
        samples = ((torch.arange(1, n + 1).view(-1, 1) - 1) + torch.rand(n, self.d)) / n
        if self.scramble:
            samples = samples[torch.randperm(n)]
        lb = self.scale[0, :]
        ub = self.scale[1, :]
        samples = lb + samples * (ub - lb)

        return samples
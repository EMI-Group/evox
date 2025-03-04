import torch
import torch.nn.functional as F

from evox.core import Algorithm, Mutable, Parameter


class DES(Algorithm):
    """The implementation of the DES algorithm.

    Reference:
    Discovering Evolution Strategies via Meta-Black-Box Optimization
    (https://arxiv.org/abs/2211.11260)

    This code has been inspired by or utilizes the algorithmic implementation from evosax.
    More information about evosax can be found at the following URL:
    GitHub Link: https://github.com/RobertTLange/evosax
    """

    def __init__(
        self,
        pop_size: int,
        center_init: torch.Tensor,
        temperature: float = 12.5,
        sigma_init: float = 0.1,
        device: torch.device | None = None,
    ):
        """Initialize the DES algorithm with the given parameters.

        :param pop_size: The size of the population.
        :param center_init: The initial center of the population. Must be a 1D tensor.
        :param temperature: The temperature parameter for the softmax. Defaults to 12.5.
        :param sigma_init: The initial standard deviation of the noise. Defaults to 0.1.
        :param device: The device to use for the tensors. Defaults to None.
        """
        super().__init__()
        assert pop_size > 1
        dim = center_init.shape[0]
        # set hyperparameters
        self.temperature = Parameter(temperature, device=device)
        self.sigma_init = Parameter(sigma_init, device=device)
        self.lrate_mean = Parameter(1.0, device=device)
        self.lrate_sigma = Parameter(0.1, device=device)
        # set value
        ranks = torch.arange(pop_size, device=device) / (pop_size - 1) - 0.5
        self.dim = dim
        self.ranks = ranks
        self.pop_size = pop_size
        # setup
        center_init = center_init.to(device=device)
        self.center = Mutable(center_init)
        self.sigma = Mutable(sigma_init * torch.ones(self.dim, device=device))

    def step(self):
        """Step the DES algorithm by sampling the population, evaluating the fitness, and updating the center."""
        device = self.center.device

        noise = torch.randn(self.pop_size, self.dim, device=device)
        population = self.center + noise * self.sigma

        fitness = self.evaluate(population)

        population = population[fitness.argsort()]

        weight = F.softmax(-20 * F.sigmoid(self.temperature * self.ranks), dim=0)
        weight = torch.tile(weight[:, None], (1, self.dim))

        weight_mean = (weight * population).sum(dim=0)
        weight_sigma = torch.sqrt((weight * (population - self.center) ** 2).sum(dim=0) + 1e-6)

        center = self.center + self.lrate_mean * (weight_mean - self.center)
        sigma = self.sigma + self.lrate_sigma * (weight_sigma - self.sigma)

        self.center = center
        self.sigma = sigma

    def record_step(self):
        return {
            "center": self.center,
            "sigma": self.sigma,
        }

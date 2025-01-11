import torch
import torch.nn.functional as F

from ...core import Algorithm, Mutable, Parameter, jit_class


@jit_class
class DES(Algorithm):
    def __init__(
        self,
        pop_size: int,
        center_init: torch.Tensor,
        temperature: float = 12.5,
        sigma_init: float = 0.1,
        mean_decay: float = 0.0,
        device: torch.device | None = None,
    ):
        super().__init__()

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
        device = self.center.device

        noise = torch.randn(self.pop_size, self.dim, device=device)
        popluation = self.center + noise * self.sigma

        fitness = self.evaluate(popluation)

        popluation = popluation[fitness.argsort()]

        weight = F.softmax(-20 * F.sigmoid(self.temperature * self.ranks), dim=0)
        weight = torch.tile(weight[:, torch.newaxis], (1, self.dim))

        weight_mean = (weight * popluation).sum(dim=0)
        weight_sigma = torch.sqrt((weight * (popluation - self.center) ** 2).sum(dim=0) + 1e-6)

        center = self.center + self.lrate_mean * (weight_mean - self.center)
        sigma = self.sigma + self.lrate_sigma * (weight_sigma - self.sigma)

        self.center = center
        self.sigma = sigma

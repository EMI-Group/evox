import math

import torch

from evox.core import Algorithm, Mutable, Parameter


class XNES(Algorithm):
    """The implementation of the xNES algorithm.

    Reference:
    Exponential Natural Evolution Strategies
    (https://dl.acm.org/doi/abs/10.1145/1830483.1830557)
    """

    def __init__(
        self,
        init_mean: torch.Tensor,
        init_covar: torch.Tensor,
        pop_size: int | None = None,
        recombination_weights: torch.Tensor | None = None,
        learning_rate_mean: float | None = None,
        learning_rate_var: float | None = None,
        learning_rate_B: float | None = None,
        covar_as_cholesky: bool = False,
        device: torch.device | None = None,
    ):
        """Initialize the xNES algorithm with the given parameters.

        :param init_mean: The initial mean vector of the population. Must be a 1D tensor.
        :param init_covar: The initial covariance matrix of the population. Must be a 2D tensor.
        :param pop_size: The size of the population. Defaults to None.
        :param recombination_weights: The recombination weights of the population. Defaults to None.
        :param learning_rate_mean: The learning rate for the mean vector. Defaults to None.
        :param learning_rate_var: The learning rate for the variance vector. Defaults to None.
        :param learning_rate_B: The learning rate for the B matrix. Defaults to None.
        :param covar_as_cholesky: Whether to use the covariance matrix as a Cholesky factorization result. Defaults to False.
        :param device: The device to use for the tensors. Defaults to None.
        """
        super().__init__()
        dim = init_mean.shape[0]
        if pop_size is None:
            pop_size = 4 + math.floor(3 * math.log(self.dim))
        assert pop_size > 0

        if learning_rate_mean is None:
            learning_rate_mean = 1
        if learning_rate_var is None:
            learning_rate_var = (9 + 3 * math.log(dim)) / 5 / math.pow(dim, 1.5)
        if learning_rate_B is None:
            learning_rate_B = learning_rate_var
        assert learning_rate_mean > 0 and learning_rate_var > 0 and learning_rate_B > 0

        if not covar_as_cholesky:
            init_covar = torch.linalg.cholesky(init_covar)

        if recombination_weights is None:
            recombination_weights = torch.arange(1, pop_size + 1)
            recombination_weights = torch.clip(math.log(pop_size / 2 + 1) - torch.log(recombination_weights), 0)
            recombination_weights = recombination_weights / torch.sum(recombination_weights) - 1 / pop_size
        assert (
            recombination_weights[1:] <= recombination_weights[:-1]
        ).all(), "recombination_weights must be in descending order"

        # set hyperparameters
        self.learning_rate_mean = Parameter(learning_rate_mean, device=device)
        self.learning_rate_var = Parameter(learning_rate_var, device=device)
        self.learning_rate_B = Parameter(learning_rate_B, device=device)
        # set value
        recombination_weights = recombination_weights.to(device=device)
        self.dim = dim
        self.pop_size = pop_size
        self.recombination_weights = recombination_weights
        # setup
        init_mean = init_mean.to(device=device)
        init_covar = init_covar.to(device=device)
        sigma = torch.pow(torch.prod(torch.diag(init_covar)), 1 / self.dim)
        self.sigma = Mutable(sigma)
        self.mean = Mutable(init_mean)
        self.B = Mutable(init_covar / sigma)

    def step(self):
        """Run one step of the xNES algorithm.

        The function will sample a population, evaluate their fitness, and then
        update the center and covariance of the algorithm using the sampled
        population.
        """
        pass
        device = self.mean.device

        noise = torch.randn(self.pop_size, self.dim, device=device)
        population = self.mean + self.sigma * (noise @ self.B.T)

        fitness = self.evaluate(population)

        order = torch.argsort(fitness)
        fitness, noise = fitness[order], noise[order]

        weights = self.recombination_weights

        Ind = torch.eye(self.dim, device=device)

        grad_delta = torch.sum(weights[:, None] * noise, dim=0)
        grad_M = (weights * noise.T) @ noise - torch.sum(weights) * Ind
        grad_sigma = torch.trace(grad_M) / self.dim
        grad_B = grad_M - grad_sigma * Ind

        mean = self.mean + self.learning_rate_mean * self.sigma * self.B @ grad_delta
        sigma = self.sigma * torch.exp(self.learning_rate_var / 2 * grad_sigma)
        B = self.B @ torch.linalg.matrix_exp(self.learning_rate_B / 2 * grad_B)

        self.sigma = sigma
        self.mean = mean
        self.B = B

    def record_step(self):
        return {"mean": self.mean, "sigma": self.sigma, "B": self.B}


class SeparableNES(Algorithm):
    """The implementation of the Separable NES algorithm.

    Reference:
    Natural Evolution Strategies
    (https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf)
    """

    def __init__(
        self,
        init_mean: torch.Tensor,
        init_std: torch.Tensor,
        pop_size: int | None = None,
        recombination_weights: torch.Tensor | None = None,
        learning_rate_mean: float | None = None,
        learning_rate_var: float | None = None,
        device: torch.device | None = None,
    ):
        """Initialize the Separable NES algorithm with the given parameters.

        :param init_mean: The initial mean vector of the population. Must be a 1D tensor.
        :param init_std: The initial standard deviation for each dimension. Must be a 1D tensor.
        :param pop_size: The size of the population. Defaults to None.
        :param recombination_weights: The recombination weights of the population. Defaults to None.
        :param learning_rate_mean: The learning rate for the mean vector. Defaults to None.
        :param learning_rate_var: The learning rate for the variance vector. Defaults to None.
        :param device: The device to use for the tensors. Defaults to None.
        """
        super().__init__()
        dim = init_mean.shape[0]
        assert init_std.shape == (dim,)

        if pop_size is None:
            pop_size = 4 + math.floor(3 * math.log(self.dim))
        assert pop_size > 0

        if learning_rate_mean is None:
            learning_rate_mean = 1
        if learning_rate_var is None:
            learning_rate_var = (3 + math.log(dim)) / 5 / math.sqrt(dim)
        assert learning_rate_mean > 0 and learning_rate_var > 0

        if recombination_weights is None:
            recombination_weights = torch.arange(1, pop_size + 1)
            recombination_weights = torch.clip(math.log(pop_size / 2 + 1) - torch.log(recombination_weights), 0)
            recombination_weights = recombination_weights / torch.sum(recombination_weights) - 1 / pop_size
        assert recombination_weights.shape == (pop_size,)

        # set hyperparameters
        self.learning_rate_mean = Parameter(learning_rate_mean, device=device)
        self.learning_rate_var = Parameter(learning_rate_var, device=device)
        # set value
        recombination_weights = recombination_weights.to(device=device)
        self.dim = dim
        self.pop_size = pop_size
        self.weight = recombination_weights
        # setup
        init_std = init_std.to(device=device)
        init_mean = init_mean.to(device=device)
        self.mean = Mutable(init_mean)
        self.sigma = Mutable(init_std)

    def step(self):
        """Run one step of the Separable NES algorithm.

        The function will sample a population, evaluate their fitness, and then
        update the center and covariance of the algorithm using the sampled
        population.
        """
        device = self.mean.device

        zero_mean_pop = torch.randn(self.pop_size, self.dim, device=device)
        population = self.mean + zero_mean_pop * self.sigma

        fitness = self.evaluate(population)

        order = torch.argsort(fitness)
        fitness, population, zero_mean_pop = fitness[order], population[order], zero_mean_pop[order]

        weight = torch.tile(self.weight[:, None], (1, self.dim))

        grad_μ = torch.sum(weight * zero_mean_pop, dim=0)
        grad_sigma = torch.sum(weight * (zero_mean_pop * zero_mean_pop - 1), dim=0)

        mean = self.mean + self.learning_rate_mean * self.sigma * grad_μ
        sigma = self.sigma * torch.exp(self.learning_rate_var / 2 * grad_sigma)

        self.mean = mean
        self.sigma = sigma

    def record_step(self):
        return {"mean": self.mean, "sigma": self.sigma}

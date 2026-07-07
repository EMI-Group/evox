from typing import Literal

import torch

from evox.core import Algorithm, Mutable, Parameter
from evox.triton_kernels.kernels.virtual_noise import (
    compute_offsets,
    virtual_bias_gradient,
    virtual_weight_gradient,
)

from .adam_step import adam_single_tensor


class VirtualES(Algorithm):
    """Virtual-population Evolution Strategy for neuroevolution.

    Instead of materializing a full ``(pop_size, dim)`` population, this algorithm
    stores a center vector ``(dim,)`` and per-individual seeds ``(pop_size,)``.
    Per-individual full weight/bias Gaussian perturbations are generated on-demand
    via the fused virtual-noise kernel, achieving O(dim) memory instead of
    O(pop_size * dim) while still searching in the full parameter space.

    The algorithm passes a ``(center_flat, seeds, sigma)`` tuple to
    ``self.evaluate()``, which the workflow routes to a virtual-population problem
    for on-demand evaluation.
    """

    def __init__(
        self,
        param_shapes: list[tuple],
        pop_size: int,
        center_init: torch.Tensor,
        learning_rate: float,
        noise_stdev: float,
        optimizer: Literal["adam"] | None = None,
        device: torch.device | None = None,
    ):
        """Initialize the VirtualES algorithm with the given parameters.

        :param param_shapes: List of parameter shapes, e.g. [(256, 784), (256,), (10, 256), (10,)].
        :param pop_size: The size of the population.
        :param center_init: The initial center of the population. Must be a 1D tensor of length
            equal to the total number of parameters (sum of element counts over param_shapes).
        :param learning_rate: The learning rate for the optimizer.
        :param noise_stdev: The standard deviation of the noise (sigma).
        :param optimizer: The optimizer to use. Defaults to None. Currently, only "adam" or None is supported.
        :param device: The device to use for the tensors. Defaults to None.
        """
        super().__init__()
        device = torch.get_default_device() if device is None else device
        assert noise_stdev > 0, "noise_stdev must be greater than 0."
        assert learning_rate > 0, "learning_rate must be greater than 0."
        assert pop_size > 0, "pop_size must be greater than 0."
        assert optimizer in [None, "adam"], "optimizer must be None or 'adam'."
        # set hyperparameters
        self.learning_rate = Parameter(learning_rate, device=device)
        self.noise_stdev = Parameter(noise_stdev, device=device)
        # set value
        center_init = center_init.to(device=device)
        self.param_shapes = param_shapes
        self.pop_size = pop_size
        self.dim = sum(torch.tensor(s, device=device).prod().item() for s in param_shapes)
        self.offsets = compute_offsets(param_shapes)
        self.optimizer = optimizer
        assert center_init.shape[0] == self.dim, (
            "center_init length must equal the total number of parameters (sum of element counts over param_shapes)."
        )
        # setup
        self.center = Mutable(center_init)
        self.seeds = Mutable(torch.randint(0, 2**31, (pop_size,), device=device, dtype=torch.int64))
        # TODO: use submodule instead of string for optimizer in the future
        if optimizer == "adam":
            self.exp_avg = Mutable(torch.zeros_like(self.center))
            self.exp_avg_sq = Mutable(torch.zeros_like(self.center))
            self.beta1 = Parameter(0.9, device=device)
            self.beta2 = Parameter(0.999, device=device)

    def step(self):
        """Step the VirtualES algorithm by regenerating seeds, evaluating fitness
        through the virtual-population tuple, and updating the center."""
        device = self.center.device

        # 1. Regenerate seeds for this generation (new random perturbation directions)
        self.seeds = torch.randint(0, 2**31, (self.pop_size,), device=device, dtype=torch.int64)

        # 2. Evaluate: pass (center, seeds, sigma) tuple through the workflow.
        #    sigma must be a Python float; extract it from the Parameter tensor if needed.
        sigma = self.noise_stdev.item() if isinstance(self.noise_stdev, torch.Tensor) else self.noise_stdev
        fitness = self.evaluate((self.center, self.seeds, sigma))

        # 3. Compute gradient from virtual noise + fitness
        flat_grad_parts = []
        for shape, offset in zip(self.param_shapes, self.offsets):
            if len(shape) == 2:
                grad = virtual_weight_gradient(fitness, self.seeds, list(shape), sigma, self.pop_size, offset)
            else:
                grad = virtual_bias_gradient(fitness, self.seeds, list(shape), sigma, self.pop_size, offset)
            flat_grad_parts.append(grad.reshape(-1))
        flat_grad = torch.cat(flat_grad_parts)

        # 4. Update center
        if self.optimizer is None:
            center = self.center - self.learning_rate * flat_grad
        else:
            center, self.exp_avg, self.exp_avg_sq = adam_single_tensor(
                self.center,
                flat_grad,
                self.exp_avg,
                self.exp_avg_sq,
                self.beta1,
                self.beta2,
                self.learning_rate,
            )
        self.center = center

    def record_step(self):
        return {"center": self.center}


# Backward-compatible alias.
VirtualLoRAES = VirtualES

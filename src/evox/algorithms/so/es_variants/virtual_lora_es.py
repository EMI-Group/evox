from typing import Literal

import torch

from evox.core import Algorithm, Mutable, Parameter
from evox.triton_kernels.kernels.lora_noise import (
    compute_counter_offsets,
    generate_lora_factors,
    lora_gradient,
)

from .adam_step import adam_single_tensor


class VirtualLoRAES(Algorithm):
    """Virtual LoRA-based Evolution Strategy for neuroevolution.

    Instead of materializing a full (pop_size, dim) population, this algorithm
    stores a center vector (dim,) and per-individual seeds (pop_size,). Using
    the Philox counter-based PRNG and LoRA low-rank noise, per-individual weight
    perturbations are generated on-demand, achieving O(dim) memory instead of
    O(pop_size * dim).

    For a 2D weight (d, k), rather than generating a full (d, k) noise matrix per
    individual, two low-rank factors A (rank, k) and B (d, rank) are sampled and
    combined as ``delta_W = B @ A``, reducing the search-space dimensionality from
    ``d * k`` to ``rank * (d + k)``.

    The algorithm passes a ``(center_flat, seeds, sigma)`` tuple to
    ``self.evaluate()``, which the workflow routes to a ``VirtualLoRAProblem``
    for on-demand evaluation.
    """

    def __init__(
        self,
        param_shapes: list[tuple],
        lora_rank: int,
        pop_size: int,
        center_init: torch.Tensor,
        learning_rate: float,
        noise_stdev: float,
        optimizer: Literal["adam"] | None = None,
        device: torch.device | None = None,
    ):
        """Initialize the VirtualLoRAES algorithm with the given parameters.

        :param param_shapes: List of parameter shapes, e.g. [(256, 784), (256,), (10, 256), (10,)].
        :param lora_rank: LoRA rank for 2D weights (e.g. 4, 8, 16).
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
        assert lora_rank > 0, "lora_rank must be greater than 0."
        assert optimizer in [None, "adam"], "optimizer must be None or 'adam'."
        # set hyperparameters
        self.learning_rate = Parameter(learning_rate, device=device)
        self.noise_stdev = Parameter(noise_stdev, device=device)
        # set value
        center_init = center_init.to(device=device)
        self.param_shapes = param_shapes
        self.lora_rank = lora_rank
        self.pop_size = pop_size
        self.dim = sum(torch.tensor(s, device=device).prod().item() for s in param_shapes)
        self.counter_offsets = compute_counter_offsets(param_shapes, lora_rank)
        self.optimizer = optimizer
        assert center_init.shape[0] == self.dim, (
            "center_init length must equal the total number of parameters (sum of element counts over param_shapes)."
        )
        # setup
        self.center = Mutable(center_init)
        self.seeds = Mutable(
            torch.randint(0, 2**31, (pop_size,), device=device, dtype=torch.int64)
        )
        # TODO: use submodule instead of string for optimizer in the future
        if optimizer == "adam":
            self.exp_avg = Mutable(torch.zeros_like(self.center))
            self.exp_avg_sq = Mutable(torch.zeros_like(self.center))
            self.beta1 = Parameter(0.9, device=device)
            self.beta2 = Parameter(0.999, device=device)

    def step(self):
        """Step the VirtualLoRAES algorithm by regenerating seeds, generating LoRA factors,
        evaluating fitness through the virtual-population tuple, and updating the center."""
        device = self.center.device

        # 1. Regenerate seeds for this generation (new random perturbation directions)
        self.seeds = torch.randint(0, 2**31, (self.pop_size,), device=device, dtype=torch.int64)

        # 2. Generate LoRA factors for each parameter block (deterministic from seeds).
        #    These same factors are used by the problem for evaluation.
        factors_per_block = []
        for shape, offset in zip(self.param_shapes, self.counter_offsets):
            factors = generate_lora_factors(self.seeds, shape, self.lora_rank, offset)
            factors_per_block.append(factors)

        # 3. Evaluate: pass (center, seeds, sigma) tuple through the workflow.
        #    sigma must be a Python float; extract it from the Parameter tensor if needed.
        sigma = self.noise_stdev.item() if isinstance(self.noise_stdev, torch.Tensor) else self.noise_stdev
        fitness = self.evaluate((self.center, self.seeds, sigma))

        # 4. Compute gradient from LoRA factors + fitness
        flat_grad_parts = []
        for factors, shape in zip(factors_per_block, self.param_shapes):
            if isinstance(factors, tuple):
                A, B = factors
                grad = lora_gradient(fitness, A, B, self.pop_size, self.noise_stdev, shape)
            else:
                # 1D weight: factors is the flat noise tensor
                grad = lora_gradient(fitness, factors, None, self.pop_size, self.noise_stdev, shape)
            flat_grad_parts.append(grad.reshape(-1))
        flat_grad = torch.cat(flat_grad_parts)

        # 5. Update center
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

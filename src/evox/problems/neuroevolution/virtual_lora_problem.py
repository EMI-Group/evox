__all__ = ["VirtualLoRAProblem"]

import warnings
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evox.core import Problem
from evox.triton_kernels import (
    compute_counter_offsets,
    generate_lora_factors,
    lora_delta_output,
)
from evox.utils import ParamsAndVector

# Activation modules supported by the layer-by-layer virtual forward pass.
# All of these apply element-wise and are therefore compatible with the
# ``(pop_size, batch, features)`` activation tensors used internally.
_SUPPORTED_ACTIVATIONS = (
    nn.ReLU,
    nn.Tanh,
    nn.Sigmoid,
    nn.GELU,
    nn.LeakyReLU,
    nn.ELU,
    nn.Softmax,
    nn.Identity,
)


class VirtualLoRAProblem(Problem):
    """Virtual LoRA-based neuroevolution problem.

    Evaluates a population of LoRA-perturbed neural networks without materializing
    the full perturbed population. For each individual, the weight perturbation is
    generated on-demand from a seed using the Philox PRNG + LoRA low-rank
    factorization.

    Instead of receiving a full ``(pop_size, dim)`` population, the
    :meth:`evaluate` method receives a tuple ``(center_flat, seeds, sigma)`` and
    applies LoRA perturbations layer-by-layer during the forward pass. Activations
    are kept at ``(pop_size, batch, features)`` — far smaller than the full
    perturbed population ``(pop_size, total_params)``.

    Supports ``nn.Sequential`` models containing ``nn.Linear`` layers and common
    activation functions (ReLU, Tanh, Sigmoid, GELU, LeakyReLU, ELU, Softmax,
    Identity).

    ```{warning}
    This problem does NOT support HPO wrapper (``problems.hpo_wrapper.HPOProblemWrapper``).
    ```
    """

    def __init__(
        self,
        model: nn.Sequential,
        data_loader: DataLoader,
        criterion: nn.Module,
        lora_rank: int,
        n_batch_per_eval: int = 1,
        device: torch.device | None = None,
        reduction: str = "mean",
    ):
        """Initialize the ``VirtualLoRAProblem``.

        :param model: The neural network model. Must be an ``nn.Sequential`` whose
            children are ``nn.Linear`` layers and/or supported activation modules.
        :param data_loader: The data loader providing the dataset for evaluation.
        :param criterion: The loss function used to evaluate the parameters'
            performance. Use ``reduction='none'`` so that per-sample losses are
            returned and this problem handles the aggregation. If a scalar
            criterion is provided, a warning is emitted.
        :param lora_rank: The LoRA rank ``r`` used for the low-rank perturbation
            factorization.
        :param n_batch_per_eval: The number of batches to be evaluated in a single
            evaluation. When set to -1, will go through the whole dataset.
            Defaults to 1.
        :param device: The device to run the computations on. Defaults to the
            current default device.
        :param reduction: The reduction method for aggregating per-sample losses
            across the batch (and across multiple batches). ``'mean'`` | ``'sum'``.
            Defaults to ``'mean'``.
        """
        super().__init__()
        if not isinstance(model, nn.Sequential):
            raise TypeError(
                f"VirtualLoRAProblem requires an nn.Sequential model, got {type(model).__name__}."
            )
        self.device = torch.get_default_device() if device is None else device

        # Store configuration.
        self.model = model
        self.data_loader = data_loader
        self.data_loader_iter = iter(data_loader)
        self.criterion = criterion
        self.lora_rank = lora_rank
        self.n_batch_per_eval = n_batch_per_eval
        self.reduction = reduction

        # 1. Extract parameter info from the model.
        params: Dict[str, nn.Parameter] = dict(model.named_parameters())
        self.param_names: List[str] = list(params.keys())
        self.param_shapes: List[Tuple[int, ...]] = [tuple(p.shape) for p in params.values()]
        self.counter_offsets: List[int] = compute_counter_offsets(self.param_shapes, lora_rank)

        # 2. ParamsAndVector for flat vector <-> param dict conversion.
        self.params_and_vector = ParamsAndVector(model)

        # 3. Build a mapping from Linear layer module names to their weight/bias
        #    parameter indices, for fast lookup during the virtual forward pass.
        self.linear_param_indices: Dict[str, Dict[str, int]] = {}
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                indices: Dict[str, int] = {}
                weight_key = f"{name}.weight"
                if weight_key in self.param_names:
                    indices["weight"] = self.param_names.index(weight_key)
                if module.bias is not None:
                    bias_key = f"{name}.bias"
                    if bias_key in self.param_names:
                        indices["bias"] = self.param_names.index(bias_key)
                self.linear_param_indices[name] = indices

    def _virtual_forward(
        self,
        inputs: torch.Tensor,
        center_params: Dict[str, nn.Parameter],
        seeds: torch.Tensor,
        sigma: float,
        pop_size: int,
    ) -> torch.Tensor:
        """Layer-by-layer forward pass with on-demand LoRA perturbations.

        For each layer:

        - ``nn.Linear``: compute the base output (center weight) plus the LoRA
          delta (from per-individual low-rank factors). Both weight and bias are
          perturbed. The weight delta uses the low-rank factorization
          ``sigma * (h @ A^T) @ B^T`` (via :func:`lora_delta_output`), which
          avoids materializing the ``(out_features, in_features)`` weight delta.
          The bias delta is direct Gaussian noise (1-D weight).
        - Activation modules: applied element-wise.

        Activations are ``(pop_size, batch, features)`` — much smaller than the
        full perturbed population ``(pop_size, total_params)``.

        :param inputs: Input batch of shape ``(batch, in_features)``.
        :param center_params: Parameter dict of the center (un-perturbed) model.
        :param seeds: 1-D ``int64`` tensor of per-individual seeds ``(pop_size,)``.
        :param sigma: Noise standard deviation.
        :param pop_size: The population size (number of individuals).
        :return: Output activations of shape ``(pop_size, batch, out_features)``.
        """
        batch_size = inputs.shape[0]
        # Expand input for all individuals: (pop_size, batch, in_features).
        h = inputs.unsqueeze(0).expand(pop_size, batch_size, -1).contiguous().float()

        for name, module in self.model.named_children():
            if isinstance(module, nn.Linear):
                # Get center weight and bias from the center param dict.
                weight = center_params[f"{name}.weight"]  # (out_features, in_features)
                w_shape = tuple(weight.shape)

                # Base output: (pop_size, batch, out_features)
                base = h @ weight.t()
                if module.bias is not None:
                    bias = center_params[f"{name}.bias"]  # (out_features,)
                    base = base + bias

                # LoRA delta for weight: sigma * (h @ A^T) @ B^T
                weight_idx = self.linear_param_indices[name]["weight"]
                A, B = generate_lora_factors(seeds, w_shape, self.lora_rank, self.counter_offsets[weight_idx])
                # A: (pop_size, rank, in_features), B: (pop_size, out_features, rank)
                delta = lora_delta_output(h, A, B, sigma)  # (pop_size, batch, out_features)
                result = base + delta

                # LoRA delta for bias (1-D — direct noise, no low-rank factorization).
                if module.bias is not None:
                    bias_idx = self.linear_param_indices[name]["bias"]
                    b_shape = tuple(bias.shape)
                    bias_noise = generate_lora_factors(
                        seeds, b_shape, self.lora_rank, self.counter_offsets[bias_idx]
                    )
                    # bias_noise: (pop_size, out_features) — direct Gaussian noise.
                    result = result + sigma * bias_noise.unsqueeze(1)  # broadcast over batch

                h = result

            elif isinstance(module, _SUPPORTED_ACTIVATIONS):
                # Element-wise activation: works directly with (pop_size, batch, features).
                h = module(h)
            else:
                raise NotImplementedError(
                    f"Layer type {type(module).__name__} is not supported by VirtualLoRAProblem. "
                    f"Supported types: nn.Linear and common activation modules "
                    f"(ReLU, Tanh, Sigmoid, GELU, LeakyReLU, ELU, Softmax, Identity)."
                )

        return h  # (pop_size, batch, out_features)

    def evaluate(self, payload) -> torch.Tensor:
        """Evaluate the fitness of a virtual LoRA-perturbed population.

        :param payload: A tuple ``(center_flat, seeds, sigma)`` where:

            - ``center_flat``: ``(dim,)`` float32 tensor — flat center vector.
            - ``seeds``: ``(pop_size,)`` int64 tensor — per-individual seeds.
            - ``sigma``: Python float — noise standard deviation.

        :return: A tensor of shape ``(pop_size,)`` containing the fitness of each
            individual.
        """
        center_flat, seeds, sigma = payload
        pop_size = seeds.shape[0]

        # Convert flat center to param dict.
        center_params = self.params_and_vector.to_params(center_flat)

        # Evaluate over the requested number of batches.
        per_individual_losses = []
        if self.n_batch_per_eval == -1:
            batches = list(self.data_loader)
        else:
            batches = []
            for _ in range(self.n_batch_per_eval):
                try:
                    batches.append(next(self.data_loader_iter))
                except StopIteration:
                    self.data_loader_iter = iter(self.data_loader)
                    batches.append(next(self.data_loader_iter))

        for inputs, labels in batches:
            inputs = inputs.to(device=self.device, non_blocking=True)
            labels = labels.to(device=self.device, non_blocking=True)

            # Forward pass with virtual LoRA population.
            # logits: (pop_size, batch, out_features)
            logits = self._virtual_forward(inputs, center_params, seeds, sigma, pop_size)

            # Compute per-sample loss. Reshape for the criterion so that it works
            # with all criteria regardless of whether they broadcast over the
            # leading pop_size dimension.
            batch_size = inputs.shape[0]
            flat_logits = logits.reshape(pop_size * batch_size, -1)
            flat_labels = labels.unsqueeze(0).expand(pop_size, -1).reshape(-1)
            flat_loss = self.criterion(flat_logits, flat_labels)

            if flat_loss.ndim == 0:
                # Criterion produced a single scalar (e.g. reduction='mean').
                # The reduction over the batch is already done by the criterion.
                warnings.warn(
                    "Criterion output is a scalar. We recommend setting `reduction` to 'none' "
                    "for the criterion and letting VirtualLoRAProblem handle the reduction."
                )
                per_individual_loss = flat_loss.expand(pop_size)
            else:
                per_sample_loss = flat_loss.reshape(pop_size, batch_size)
                if self.reduction == "mean":
                    per_individual_loss = per_sample_loss.mean(dim=1)  # (pop_size,)
                else:  # "sum"
                    per_individual_loss = per_sample_loss.sum(dim=1)  # (pop_size,)
            per_individual_losses.append(per_individual_loss)

        # Aggregate over batches.
        losses = torch.stack(per_individual_losses, dim=1)  # (pop_size, n_batches)
        if self.reduction == "mean":
            fitness = losses.mean(dim=1)
        else:  # "sum"
            fitness = losses.sum(dim=1)
        return fitness  # (pop_size,)

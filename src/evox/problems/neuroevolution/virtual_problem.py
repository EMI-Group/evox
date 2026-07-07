__all__ = ["VirtualProblem"]

import warnings
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evox.core import Problem
from evox.triton_kernels import (
    compute_offsets,
    virtual_perturbed_linear,
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


class VirtualProblem(Problem):
    """Virtual Gaussian-noise neuroevolution problem.

    Evaluates a population of Gaussian-noise-perturbed neural networks without
    materializing the full perturbed population. For each individual, the weight
    and bias perturbations are generated on-demand from a seed using the
    fused kernel :func:`virtual_perturbed_linear`.

    Instead of receiving a full ``(pop_size, dim)`` population, the
    :meth:`evaluate` method receives a tuple ``(center_flat, seeds, sigma)`` and
    applies full Gaussian-noise perturbations layer-by-layer during the forward
    pass via :func:`virtual_perturbed_linear`, which does the base matmul plus
    ``sigma * noise`` perturbation of both weight and bias in a single call.
    Activations are kept at ``(pop_size, batch, features)`` — far smaller than
    the full perturbed population ``(pop_size, total_params)``.

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
        n_batch_per_eval: int = 1,
        device: torch.device | None = None,
        reduction: str = "mean",
    ):
        """Initialize the ``VirtualProblem``.

        :param model: The neural network model. Must be an ``nn.Sequential`` whose
            children are ``nn.Linear`` layers and/or supported activation modules.
        :param data_loader: The data loader providing the dataset for evaluation.
        :param criterion: The loss function used to evaluate the parameters'
            performance. Use ``reduction='none'`` so that per-sample losses are
            returned and this problem handles the aggregation. If a scalar
            criterion is provided, a warning is emitted.
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
            raise TypeError(f"VirtualProblem requires an nn.Sequential model, got {type(model).__name__}.")
        self.device = torch.get_default_device() if device is None else device

        # Store configuration.
        self.model = model
        self.data_loader = data_loader
        self.data_loader_iter = iter(data_loader)
        self.criterion = criterion
        self.n_batch_per_eval = n_batch_per_eval
        self.reduction = reduction

        # 1. Extract parameter info from the model.
        params: Dict[str, nn.Parameter] = dict(model.named_parameters())
        self.param_names: List[str] = list(params.keys())
        self.param_shapes: List[Tuple[int, ...]] = [tuple(p.shape) for p in params.values()]
        self.offsets: List[int] = compute_offsets(self.param_shapes)

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
        """Layer-by-layer forward pass with on-demand Gaussian-noise perturbations.

        For each layer:

        - ``nn.Linear``: compute the base output (center weight) plus a full
          Gaussian-noise perturbation of both weight and bias via
          :func:`virtual_perturbed_linear`. The noise is generated on-the-fly per
          individual from a seed and the block ``offset``, without materializing
          the ``(out_features, in_features)`` weight delta.
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
        # Keep 2D: (batch, in_features) — shared across all individuals.
        # virtual_perturbed_linear handles 2D input (broadcasts across pop_size
        # internally via x_is_per_individual=False). After the first Linear call,
        # h becomes (pop_size, batch, out_features) naturally.
        h = inputs.float()

        for name, module in self.model.named_children():
            if isinstance(module, nn.Linear):
                weight = center_params[f"{name}.weight"]  # (out_features, in_features)
                bias = center_params[f"{name}.bias"] if module.bias is not None else None
                # The weight block's base offset from compute_offsets.
                weight_idx = self.linear_param_indices[name]["weight"]
                offset = self.offsets[weight_idx]
                # virtual_perturbed_linear handles base matmul + sigma*noise perturbation
                # of weight AND bias (bias_offset = offset + out*in computed internally).
                h = virtual_perturbed_linear(h, weight, bias, seeds, sigma, offset)

            elif isinstance(module, _SUPPORTED_ACTIVATIONS):
                # Element-wise activation: works directly with (pop_size, batch, features).
                h = module(h)
            else:
                raise NotImplementedError(
                    f"Layer type {type(module).__name__} is not supported by VirtualProblem. "
                    f"Supported types: nn.Linear and common activation modules "
                    f"(ReLU, Tanh, Sigmoid, GELU, LeakyReLU, ELU, Softmax, Identity)."
                )

        return h  # (pop_size, batch, out_features)

    def evaluate(self, payload) -> torch.Tensor:
        """Evaluate the fitness of a virtual Gaussian-noise-perturbed population.

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

            # Forward pass with virtual perturbed population.
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
                    "for the criterion and letting VirtualProblem handle the reduction."
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


# Backward-compatible alias.
VirtualLoRAProblem = VirtualProblem

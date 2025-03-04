__all__ = ["SupervisedLearningProblem"]

import warnings
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evox.core import Problem, use_state, vmap

from .utils import get_vmap_model_state_forward


class SupervisedLearningProblem(Problem):
    """The supervised learning problem to test a model's parameters or a batch of parameters with given data and criterion."""

    def __init__(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        criterion: nn.Module,
        n_batch_per_eval: int = 1,
        pop_size: int | None = None,
        device: torch.device | None = None,
        reduction: str = "mean",
    ):
        """Initialize the `SupervisedLearningProblem`.

        :param model: The neural network model whose parameters need to be evaluated.
        :param data_loader: The data loader providing the dataset for evaluation.
        :param criterion: The loss function used to evaluate the parameters' performance.
        :param n_batch_per_eval: The number of batches to be evaluated in a single evaluation. When set to -1, will go through the whole dataset. Defaults to 1.
        :param pop_size: The size of the population (batch size of the parameters) to be evaluated. Defaults to None for single-run mode.
        :param device: The device to run the computations on. Defaults to the current default device.
        :param reduction: The reduction method for the criterion. 'mean' | 'sum'. Defaults to "mean".

        :raises RuntimeError: If the data loader contains no items.

        ## Warning
        This problem does NOT support HPO wrapper (`problems.hpo_wrapper.HPOProblemWrapper`), i.e., the workflow containing this problem CANNOT be vmapped.
        """
        super().__init__()
        self.device = torch.get_default_device() if device is None else device
        pop_size = 1 if pop_size is None else pop_size

        self.data_loader = data_loader
        self.data_loader_iter = iter(data_loader)
        self.n_batch_per_eval = n_batch_per_eval
        self.reduction = reduction

        # JITed model state forward initialization
        self.vmap_init_state, self.vmap_state_forward = get_vmap_model_state_forward(
            model,
            pop_size,
            device=device,
        )
        self.init_state = model.state_dict()
        self.state_forward = use_state(model)
        # JITed and vmapped state criterion initialization
        self.criterion = torch.compile(criterion)
        self.vmap_criterion = torch.compile(vmap(criterion, in_dims=(0, None)))

    def _vmap_forward_pass(
        self, model_state: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], data: Tuple[torch.Tensor, torch.Tensor]
    ):
        inputs, labels = data
        inputs = inputs.to(device=self.device, non_blocking=True)
        labels = labels.to(device=self.device, non_blocking=True)

        _model_state, logits = self.vmap_state_forward(
            model_state,
            inputs,
        )
        loss = self.vmap_criterion(
            logits,
            labels,
        )
        return loss

    def _forward_pass(
        self, model_state: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], data: Tuple[torch.Tensor, torch.Tensor]
    ):
        inputs, labels = data
        inputs = inputs.to(device=self.device, non_blocking=True)
        labels = labels.to(device=self.device, non_blocking=True)

        _model_state, logits = self.state_forward(
            model_state,
            inputs,
        )
        loss = self.criterion(
            logits,
            labels,
        )
        return loss

    def _vmap_evaluate(
        self,
        pop_params: Dict[str, nn.Parameter],
    ):
        # Initialize model and criterion states
        model_state = self.vmap_init_state | pop_params

        losses = []
        if self.n_batch_per_eval == -1:
            for data in self.data_loader:
                losses.append(self._vmap_forward_pass(model_state, data))
        else:
            for _ in range(self.n_batch_per_eval):
                losses.append(self._vmap_forward_pass(model_state, next(self.data_loader_iter)))
        if losses[0].ndim == 1:
            losses = torch.stack(losses, dim=1)
            warnings.warn(
                "Criterion output is a scalar. We recommend setting `reduction` to 'none' for criterion and let SupervisedLearningProblem handle the reduction."
            )
        else:
            losses = torch.cat(losses, dim=1)
        if self.reduction == "mean":
            pop_fitness = losses.mean(dim=1)
        elif self.reduction == "sum":
            pop_fitness = losses.sum(dim=1)
        return pop_fitness

    def _single_evaluate(
        self,
        params: Dict[str, nn.Parameter],
    ):
        # Initialize model and criterion states
        model_state = self.init_state | params

        losses = []
        if self.n_batch_per_eval == -1:
            for data in self.data_loader:
                losses.append(self._forward_pass(model_state, data))
        else:
            for _ in range(self.n_batch_per_eval):
                losses.append(self._forward_pass(model_state, next(self.data_loader_iter)))
        if losses[0].ndim == 0:
            losses = torch.stack(losses, dim=0)
            warnings.warn(
                "Criterion output is a scalar. We recommend setting `reduction` to 'none' for criterion and let SupervisedLearningProblem handle the reduction."
            )
        else:
            losses = torch.cat(losses, dim=0)
        if self.reduction == "mean":
            pop_fitness = losses.mean(dim=0)
        elif self.reduction == "sum":
            pop_fitness = losses.sum(dim=0)
        return pop_fitness

    def evaluate(self, pop_params: Dict[str, nn.Parameter]) -> torch.Tensor:
        """Evaluate the fitness of a population (batch) of model parameters.

        :param pop_params: A dictionary of parameters where each key is a parameter name and each value is a tensor of shape (batch_size, *param_shape) representing the batched parameters of batched models.

        :return: A tensor of shape (batch_size,) containing the fitness of each sample in the population.
        """
        pop_size = next(iter(pop_params.values())).size(0)
        if pop_size != 1:
            pop_fitness = self._vmap_evaluate(pop_params)
        else:
            pop_fitness = self._single_evaluate(pop_params)
        return pop_fitness

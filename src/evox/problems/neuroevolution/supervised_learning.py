__all__ = ["SupervisedLearningProblem"]

import weakref
from typing import Dict, Iterable, Iterator, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ...core import Problem, jit, jit_class, use_state, vmap
from .utils import get_vmap_model_state_forward

__supervised_data__: Dict[
    int, Dict[str, DataLoader | Iterable | Iterator | Tuple]
] = {}  # Cannot be a weakref.WeakValueDictionary because the values are only stored here

# cSpell:words vmapped


@jit_class
class SupervisedLearningProblem(Problem):
    """The supervised learning problem to test a model's parameters or a batch of parameters with given data and criterion."""

    def __init__(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        criterion: nn.Module,
        pop_size: int | None = None,
        device: torch.device | None = None,
    ):
        """Initialize the `SupervisedLearningProblem`.

        :param model: The neural network model whose parameters need to be evaluated.
        :param data_loader: The data loader providing the dataset for evaluation.
        :param criterion: The loss function used to evaluate the parameters' performance.
        :param pop_size: The size of the population (batch size of the parameters) to be evaluated. Defaults to None for single-run mode.
        :param device: The device to run the computations on. Defaults to the current default device.

        :raises RuntimeError: If the data loader contains no items.

        ## Warning
        This problem does NOT support HPO wrapper (`problems.hpo_wrapper.HPOProblemWrapper`), i.e., the workflow containing this problem CANNOT be vmapped.
        """
        super().__init__()
        device = torch.get_default_device() if device is None else device
        pop_size = 1 if pop_size is None else pop_size

        # Global data loader info registration
        global __supervised_data__
        instance_id = id(self)
        self._index_id_ = instance_id
        if instance_id not in __supervised_data__.keys():
            __supervised_data__[instance_id] = {
                "data_loader_ref": data_loader,
                "data_loader_iter": None,
                "data_next_cache": None,
            }
            weakref.finalize(self, __supervised_data__.pop, instance_id, None)

        try:
            dummy_inputs, dummy_labels = next(iter(data_loader))
        except StopIteration:
            raise RuntimeError(f"The `data_loader` of `{self.__class__.__name__}` must contain at least one item.")
        dummy_inputs: torch.Tensor = dummy_inputs.to(device=device)
        dummy_labels: torch.Tensor = dummy_labels.to(device=device)

        # JITed model state forward initialization
        non_vmap_result, vmap_result = get_vmap_model_state_forward(
            model,
            pop_size,
            dummy_inputs,
            check_output=lambda logits: isinstance(logits, torch.Tensor) and logits.size(0) == pop_size,
            device=device,
            get_non_vmap=True,
            check_single_output=lambda logits: isinstance(logits, torch.Tensor),
        )
        model_init_state, jit_state_forward, dummy_single_logits, param_to_state_key_map, model_buffers = non_vmap_result
        vmap_model_init_state, jit_vmap_state_forward, dummy_vmap_logits, _, _ = vmap_result
        self._jit_state_forward = jit_state_forward
        self._jit_vmap_state_forward = jit_vmap_state_forward
        self._param_to_state_key_map = param_to_state_key_map
        self._model_buffers = model_buffers

        # JITed and vmapped state criterion initialization
        state_criterion = use_state(lambda: criterion.forward)
        criterion_init_state = state_criterion.init_state()
        self._jit_state_criterion = jit(
            state_criterion,
            trace=True,
            lazy=False,
            example_inputs=(
                criterion_init_state,
                dummy_single_logits,
                dummy_labels,
            ),
        )
        vmap_state_criterion = vmap(state_criterion, in_dims=(0, 0, None))
        vmap_criterion_init_state = vmap_state_criterion.init_state(pop_size)
        self._jit_vmap_state_criterion = jit(
            vmap_state_criterion,
            trace=True,
            lazy=False,
            example_inputs=(
                vmap_criterion_init_state,
                dummy_vmap_logits,
                dummy_labels,
            ),
        )
        self.vmap_criterion_init_state = vmap_criterion_init_state

        # Model parameters and buffers registration
        self._model_buffers = {key: value for key, value in model_init_state.items() if key not in self._param_to_state_key_map}
        sample_param_key, sample_state_key = next(iter(self._param_to_state_key_map.items()))
        self._sample_param_key = sample_param_key
        self._sample_param_ndim = model_init_state[sample_state_key].ndim

        # Other member variables registration
        self.criterion_init_state = criterion_init_state

    @torch.jit.ignore
    def _data_loader_reset(self) -> None:
        global __supervised_data__
        data_info = __supervised_data__[self._index_id_]
        data_info["data_loader_iter"] = iter(data_info["data_loader_ref"])
        try:
            data_info["data_next_cache"] = next(data_info["data_loader_iter"])
        except StopIteration:
            data_info["data_next_cache"] = None

    @torch.jit.ignore
    def _data_loader_next(self) -> Tuple[torch.Tensor, torch.Tensor]:
        global __supervised_data__
        data_info = __supervised_data__[self._index_id_]
        next_data = data_info["data_next_cache"]
        try:
            data_info["data_next_cache"] = next(data_info["data_loader_iter"])
        except StopIteration:
            data_info["data_next_cache"] = None
        return next_data

    @torch.jit.ignore
    def _data_loader_has_next(self) -> bool:
        global __supervised_data__
        return __supervised_data__[self._index_id_]["data_next_cache"] is not None

    def _vmap_evaluate(
        self,
        pop_params: Dict[str, nn.Parameter],
        num_map: int,
        device: torch.device,
    ):
        # Initialize model and criterion states
        model_buffers = {  # expand dimensions for model buffers
            key: value.unsqueeze(0).expand([num_map] + list(value.shape)) for key, value in self._model_buffers.items()
        }
        state_params = {self._param_to_state_key_map[key]: value for key, value in pop_params.items()}
        model_state = model_buffers
        model_state.update(state_params)
        criterion_state = {key: value.clone() for key, value in self.vmap_criterion_init_state.items()}

        total_result = torch.zeros(num_map, device=device)
        total_inputs = 0
        self._data_loader_reset()
        while self._data_loader_has_next():
            inputs, labels = self._data_loader_next()
            inputs = inputs.to(device=device, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)

            model_state, logits = self._jit_vmap_state_forward(
                model_state,
                inputs,
            )
            criterion_state, result = self._jit_vmap_state_criterion(
                criterion_state,
                logits,
                labels,
            )
            total_result += result * inputs.size(0)
            total_inputs += inputs.size(0)
        pop_fitness = total_result / total_inputs
        return pop_fitness

    def _single_evaluate(
        self,
        params: Dict[str, nn.Parameter],
        device: torch.device,
    ):
        # Initialize model and criterion states
        model_buffers = {key: value.clone() for key, value in self._model_buffers.items()}
        params = {self._param_to_state_key_map[key]: value.squeeze(0) for key, value in params.items()}
        model_state = model_buffers
        model_state.update(params)
        criterion_state = {key: value.clone() for key, value in self.criterion_init_state.items()}

        # Calculate population fitness
        total_result = torch.tensor(0.0, device=device)
        total_inputs = 0
        self._data_loader_reset()
        while self._data_loader_has_next():
            inputs, labels = self._data_loader_next()
            inputs = inputs.to(device=device, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)

            model_state, logits = self._jit_state_forward(
                model_state,
                inputs,
            )
            criterion_state, result = self._jit_state_criterion(
                criterion_state,
                logits,
                labels,
            )
            total_result += result * inputs.size(0)
            total_inputs += inputs.size(0)
        fitness = total_result / total_inputs
        return fitness.unsqueeze(0)

    def evaluate(self, pop_params: Dict[str, nn.Parameter]) -> torch.Tensor:
        """Evaluate the fitness of a population (batch) of model parameters.

        :param pop_params: A dictionary of parameters where each key is a parameter name and each value is a tensor of shape (batch_size, *param_shape) representing the batched parameters of batched models.

        :return: A tensor of shape (batch_size,) containing the fitness of each sample in the population.
        """
        pop_params_value = pop_params[self._sample_param_key]
        assert (
            pop_params_value.ndim == self._sample_param_ndim + 1
        ), f"Expected exactly one batch dimension, got {pop_params_value.ndim - self._sample_param_ndim}"
        if pop_params_value.size(0) != 1:
            pop_fitness = self._vmap_evaluate(
                pop_params,
                pop_params_value.size(0),
                pop_params_value.device,
            )
        else:
            pop_fitness = self._single_evaluate(
                pop_params,
                pop_params_value.device,
            )
        return pop_fitness

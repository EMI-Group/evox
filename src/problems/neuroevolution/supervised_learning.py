import copy
import types
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Set, Tuple, Iterable, Iterator

from ...core import Problem, jit_class, use_state, vmap, jit
from ...core.module import assign_load_state_dict


__supervised_data__: Dict[int, Dict[str, DataLoader | Iterable | Iterator | Tuple]] = None


@jit_class
class SupervisedLearningProblem(Problem):

    def __init__(self, 
        model      : nn.Module,
        data_loader: DataLoader, 
        criterion  : nn.Module,
        pop_size   : int | None, # TODO
        device     : torch.device | None = None,
    ):
        super().__init__()
        device = torch.get_default_device() if device is None else device

        # Global data loader info registration
        global __supervised_data__
        if __supervised_data__ is None:
            __supervised_data__ = {}
        instance_id = hash(self)
        if instance_id not in __supervised_data__.keys():
            __supervised_data__[instance_id] = {
                "data_loader_ref" : data_loader,
                "data_loader_iter": None,
                "data_next_cache" : None,
            }
        try:
            dummy_inputs, dummy_labels = next(iter(data_loader))
        except StopIteration:
            raise RuntimeError(
                f"The `data_loader` of `{self.__class__.__name__}` must contain at least one item."
            )
        dummy_inputs: torch.Tensor = dummy_inputs.to(device=device)
        dummy_labels: torch.Tensor = dummy_labels.to(device=device)
    
        # Model initialization
        inference_model = copy.deepcopy(model) 
        inference_model = inference_model.to(device=device)
        for _, value in inference_model.named_parameters():
            value.requires_grad = False
        inference_model.load_state_dict = types.MethodType(assign_load_state_dict, inference_model)

        # JITed and vmapped model state forward initialization
        state_forward    = use_state(lambda: inference_model.forward)
        model_init_state = state_forward.init_state(clone=False)
        self._jit_state_forward, (_, dummy_single_logits) = jit(state_forward,
            trace = True, lazy = False, 
            example_inputs = (model_init_state, dummy_inputs),
            return_dummy_output = True, 
        )
        vmap_state_forward    = vmap(state_forward, in_dims=(0, None))
        vmap_model_init_state = vmap_state_forward.init_state(pop_size)
        self._jit_vmap_state_forward, (_, dummy_vmap_logits) = jit(vmap_state_forward, 
            trace = True, lazy = False, 
            example_inputs = (vmap_model_init_state, dummy_inputs),
            return_dummy_output = True, 
        )

        # Building map from model parameters key to model state key 
        model_params = dict(inference_model.named_parameters())
        self.param_to_state_key_map: Dict[str, str] = {
            params_key: state_key
            for state_key, state_value in model_init_state.items() 
            for params_key, params_value in model_params.items() 
            if torch.equal(state_value, params_value)
        }

        # JITed and vmapped state critierion initialization
        state_criterion      = use_state(lambda: criterion.forward)
        criterion_init_state = state_criterion.init_state()
        self._jit_state_criterion = jit(state_criterion,
            trace = True, lazy = False, 
            example_inputs = (criterion_init_state, dummy_single_logits, dummy_labels),
        )
        vmap_state_criterion      = vmap(state_criterion, in_dims=(0, 0, None))
        vmap_criterion_init_state = vmap_state_criterion.init_state(pop_size)
        self._jit_vmap_state_criterion = jit(vmap_state_criterion, 
            trace = True, lazy = False, 
            example_inputs = (vmap_criterion_init_state, dummy_vmap_logits, dummy_labels),
        )

        # Model parameters and buffers registration
        self._model_buffers = {
            key: value 
            for key, value in model_init_state.items() if key not in self.param_to_state_key_map
        }
        # TODO: refactor
        sample_param_key  = tuple(list(self.param_to_state_key_map.items())[0])
        self._sample_param_ndim = model_init_state[sample_param_key[1]].ndim
        self._sample_param_key = sample_param_key[0]

        # Other member variables registration
        self.criterion_init_state      = criterion_init_state
        self.vmap_criterion_init_state = vmap_criterion_init_state

    @torch.jit.ignore
    def _data_loader_reset(self) -> None:
        global __supervised_data__
        data_info = __supervised_data__[self._hash_id_]
        data_info["data_loader_iter"] = iter(data_info["data_loader_ref"])
        try:
            data_info["data_next_cache"] = next(data_info["data_loader_iter"])
        except StopIteration:
            data_info["data_next_cache"] = None

    @torch.jit.ignore
    def _data_loader_next(self) -> Tuple[torch.Tensor, torch.Tensor]:
        global __supervised_data__
        data_info = __supervised_data__[self._hash_id_]
        next_data = data_info["data_next_cache"]
        try:
            data_info["data_next_cache"] = next(data_info["data_loader_iter"])
        except StopIteration:
            data_info["data_next_cache"] = None
        return next_data

    @torch.jit.ignore
    def _data_loader_has_next(self) -> bool:
        global __supervised_data__
        return __supervised_data__[self._hash_id_]["data_next_cache"] is not None

    def _vmap_evaluate(self, pop_params: Dict[str, nn.Parameter], num_map: int, device: torch.device):
        # Initialize model and criterion states
        model_buffers = { # Expand dimensions for model buffers
            key: value.unsqueeze(0).expand([num_map] + list(value.shape))
            for key, value in self._model_buffers.items()
        }
        state_params = {self.param_to_state_key_map[key]: value for key, value in pop_params.items()}
        model_state  = model_buffers
        model_state.update(state_params)
        criterion_state = {key: value.clone() for key, value in self.vmap_criterion_init_state.items()}
        
        total_result = torch.zeros(num_map, device=device)
        total_inputs = 0
        self._data_loader_reset()
        while self._data_loader_has_next():
            inputs, labels = self._data_loader_next()
            inputs = inputs.to(device=device, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)

            model_state    , logits = self._jit_vmap_state_forward(model_state, inputs)
            criterion_state, result = self._jit_vmap_state_criterion(criterion_state, logits, labels)
            total_result += result * inputs.size(0)
            total_inputs += inputs.size(0)
        pop_fitness = total_result / total_inputs
        return pop_fitness

    def _single_evaluate(self, params: Dict[str, nn.Parameter], device: torch.device):
        # Initialize model and criterion states
        model_buffers = {key: value.clone() for key, value in self._model_buffers.items()}
        params = {"self." + key: value for key, value in params.items()}
        model_state = model_buffers
        model_state.update(params)
        criterion_state = {key: value.clone() for key, value in self.criterion_init_state.items()}

        # Calculate population fitness
        total_result = torch.tensor(0, device=device)
        total_inputs = 0
        self._data_loader_reset()
        while self._data_loader_has_next():
            inputs, labels = self._data_loader_next()
            inputs = inputs.to(device=device, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)

            model_state    , logits = self._jit_state_forward(model_state, inputs)
            criterion_state, result = self._jit_state_criterion(criterion_state, logits, labels)
            total_result += result * inputs.size(0)
            total_inputs += inputs.size(0)
        fitness = total_result / total_inputs
        return fitness

    def evaluate(self, pop_params: Dict[str, nn.Parameter]) -> torch.Tensor:
        pop_params_value = pop_params[self._sample_param_key]
        if pop_params_value.ndim > self._sample_param_ndim and pop_params_value.size(0) != 1:
            pop_fitness = self._vmap_evaluate(pop_params, pop_params_value.size(0), pop_params_value.device)
        else: 
            pop_fitness = self._single_evaluate(pop_params, pop_params_value.device)
        return pop_fitness 

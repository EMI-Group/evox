import copy
import types
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple

from ...core import Problem, jit_class, use_state, vmap, jit
from ...core.module import assign_load_state_dict


__data_loader__: Dict[int, DataLoader] = None # TODO: Add to __init__


@jit_class
class SupervisedLearningProblem(Problem):

    def __init__(self, data_loader: DataLoader):
        super().__init__()
        global __data_loader__
        __data_loader__ = data_loader
    
    def setup(self, model: nn.Module, criterion: nn.Module, device: torch.device | None = None):
        device = torch.get_default_device() if device is None else device
        inference_model = copy.deepcopy(model) # TODO
        inference_model = inference_model.to(device=device)
        for _, value in inference_model.named_parameters():
            value.requires_grad = False
        inference_model.load_state_dict = types.MethodType(assign_load_state_dict, inference_model)

        state_forward       = use_state(lambda: inference_model.forward)
        param_keys          = tuple(dict(inference_model.named_parameters()).keys())
        self._model_buffers = {
            key: value 
            for key, value in state_forward.init_state().items() if key not in param_keys
        }
        # FIXME
        self._example_param_ndim = (param_keys[0], inference_model.get_parameter(param_keys[0]).ndim)

        global __data_loader__
        dummy_inputs, dummy_labels = next(iter(__data_loader__))
        dummy_inputs = dummy_inputs.to(device=device)
        dummy_labels = dummy_labels.to(device=device)

        vmap_state_forward = vmap(state_forward, in_dims=(0, None))
        model_init_state   = vmap_state_forward.init_state(23) # TODO: move to user side
        self._jit_vmap_state_forward, (_, dummy_logits) = jit(vmap_state_forward, 
            trace=True, lazy=False, return_dummy_output=True, 
            example_inputs=(model_init_state, dummy_inputs)
        )
        self._jit_forward  = torch.jit.script(inference_model)

        dummy_labels = dummy_labels.unsqueeze(1).repeat(1, 10) # TODO: move to user side

        state_criterion      = use_state(lambda: criterion.forward)
        if len(state_criterion.init_state()) > 0:
            vmap_state_criterion = vmap(state_criterion, in_dims=(0, 0, None))
            criterion_init_state = vmap_state_criterion.init_state(23)
            self._jit_vmap_state_criterion = jit(vmap_state_criterion, 
                trace=True, lazy=False, 
                example_inputs=(criterion_init_state, dummy_logits, dummy_labels),
            )
        else:
            vmap_state_criterion = vmap(state_criterion, in_dims=(0, None))
            self._jit_vmap_state_criterion = jit(vmap_state_criterion, 
                trace=True, lazy=False, 
                example_inputs=(dummy_logits, dummy_labels),
            )
            criterion_init_state = None

        self._jit_criterion  = torch.jit.script(criterion)

        self.model                = inference_model
        self.criterion            = criterion
        self.criterion_init_state = criterion_init_state

    @torch.jit.export
    def _vmap_state_forward(self, 
            state : Dict[str, torch.Tensor], 
            inputs: torch.Tensor,
        ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        return self._jit_vmap_state_forward(state, inputs)

    @torch.jit.export
    def _vmap_state_criterion(self, 
            state : Dict[str, torch.Tensor] | None, 
            logits: torch.Tensor, 
            labels: torch.Tensor,
        ) -> torch.Tensor:
        # TODO: check empty state and branch
        return self._jit_vmap_state_criterion(logits, labels)

    @torch.jit.ignore
    def _evaluate_vmap(self, pop_params: Dict[str, nn.Parameter], num_map: int, device: torch.device):
        model_buffers = {
            key: value.unsqueeze(0).expand(num_map, *([-1] * value.ndim)) 
            for key, value in self._model_buffers.items()
        }
        # state = {**model_buffers, **pop_params}
        model_state = model_buffers
        pop_params = {"self." + key: val for key, val in pop_params.items()}
        model_state.update(pop_params)
        criterion_state = copy.deepcopy(self.criterion_init_state)
        
        n_inputs = 0
        result = torch.zeros(num_map, device=device)
        global __data_loader__
        for v in __data_loader__:
            inputs: torch.Tensor = v[0]
            labels: torch.Tensor = v[1]
            inputs = inputs.to(device=device, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)
            labels = labels.unsqueeze(1).repeat(1, 10) # TODO: move to user side

            model_state, logits = self._vmap_state_forward(model_state, inputs)
            n_inputs     += inputs.size(0)
            result       += self._vmap_state_criterion(criterion_state, logits, labels) * inputs.size(0)

        pop_fitness = result / n_inputs
        return pop_fitness

    @torch.jit.ignore
    def _evaluate_single(self, params: Dict[str, nn.Parameter], device: torch.device): # FIXME
        self.model.load_state_dict(params)

        result = torch.tensor(0, device=device)
        n_inputs = 0
        global __data_loader__
        for (inputs, labels) in __data_loader__:
            inputs.to(device, non_blocking=True)
            labels.to(device, non_blocking=True)

            logits    = self._jit_forward(inputs)
            n_inputs += inputs.size(0)
            result   += self._jit_criterion(logits, labels) * inputs.size(0)

        pop_fitness = result / n_inputs
        return pop_fitness

    def evaluate(self, pop_params: Dict[str, nn.Parameter]) -> torch.Tensor:
        pop_params_value = pop_params[self._example_param_ndim[0]]
        if pop_params_value.ndim > self._example_param_ndim[1] and pop_params_value.size(0) != 1: 
            # vmapped evaluation
            pop_fitness = self._evaluate_vmap(pop_params, pop_params_value.size(0), pop_params_value.device)
        else: 
            # single evaluation
            pop_fitness = self._evaluate_single(pop_params, pop_params_value.device)
        return pop_fitness 

        # # if vmapped 
        # buffers = align_vmap_tensor(buf, 0) for buf in buffers
        # model.load_state_dict(params + buffers)
        # # else
        # model.load_state_dict(params + buffers)
    
        # params = model.named_parameters()
        # buffers = model.named_buffers()
        # print(params)
    
        # adapter = TreeAndVector(params)
        # center = adapter.to_vector(tstate.params)

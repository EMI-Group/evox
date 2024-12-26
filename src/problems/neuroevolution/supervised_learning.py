import copy
import types
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable, Dict, Tuple

from ...utils import ParamsAndVector
from ...core import Problem, jit_class, use_state, vmap, jit
from ...core.module import assign_load_state_dict


__data_loader__: Dict[int, DataLoader] = None # TODO: Add to __init__


@jit_class
class SupervisedLearningProblem(Problem):

    def __init__(self,
        data_loader: DataLoader,
        criterion  : Callable,
    ):
        super().__init__()
        # self.data_loader = data_loader
        global __data_loader__
        __data_loader__ = data_loader
        self.criterion   = criterion
    
    def setup(self, model: nn.Module, adapter: ParamsAndVector):
        inference_model = copy.deepcopy(model)
        for _, value in inference_model.named_parameters():
            value.requires_grad = False
        inference_model.load_state_dict = types.MethodType(assign_load_state_dict, inference_model)

        state_forward       = use_state(lambda: inference_model.forward)
        param_keys    = tuple(dict(inference_model.named_parameters()).keys())
        self._model_buffers = {
            key: value 
            for key, value in state_forward.init_state().items() if key not in param_keys
        }
        self._example_param_ndim = (param_keys[0], inference_model.get_parameter(param_keys[0]).ndim)

        global __data_loader__
        vmap_state_forward           = vmap(state_forward, in_dims=(0, None))
        init_state = vmap_state_forward.init_state(11)
        self._jit_vmap_state_forward = jit(vmap_state_forward, 
            trace=True, lazy=False, 
            example_inputs=(init_state, next(iter(__data_loader__))[0])
        ) 
        self._jit_forward   = torch.jit.script(inference_model)
        self._jit_criterion = torch.jit.script(self.criterion)

        self.model       = inference_model

    @torch.jit.export
    def _vmap_state_forward(self, state: Dict[str, torch.Tensor], inputs: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        print(state[self._example_param_ndim[0]])
        return self._jit_vmap_state_forward(state, inputs)

    @torch.jit.export
    def _criterion(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self._jit_criterion(logits, labels)

    @torch.jit.ignore
    def _evaluate_vmap(self, pop_params: Dict[str, nn.Parameter], device: torch.device, num_map: int):
        model_buffers = {
            key: value.unsqueeze(0).expand(num_map, *([-1] * value.ndim)) 
            for key, value in self._model_buffers.items()
        }
        # state = {**model_buffers, **pop_params}
        state = model_buffers
        state.update(pop_params)
        n_inputs = 0
        result = torch.tensor(0, device=device)
        global __data_loader__
        for (inputs, labels) in __data_loader__:
            inputs.to(device=device, non_blocking=True)
            labels.to(device=device, non_blocking=True)

            state, logits = self._vmap_state_forward(state, inputs)
            n_inputs     += inputs.size(0)
            result       += self._criterion(logits, labels) * inputs.size(0)

        pop_fitness = result / n_inputs
        return pop_fitness

    @torch.jit.ignore
    def _evaluate_single(self, params: Dict[str, nn.Parameter], device: torch.device):
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
            pop_fitness = self._evaluate_vmap(pop_params, pop_params_value.device, pop_params_value.size(0))
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

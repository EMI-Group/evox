import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable, Dict

from ...utils import ParamsAndVector
from ...core import Problem, jit_class, use_state, vmap, jit


@jit_class
class SupervisedLearningProblem(Problem):

    def __init__(self,
        model      : nn.Module,
        data_loader: DataLoader,
        criterion  : Callable,
        adapter    : ParamsAndVector, # TODO: add a base class
    ):
        super().__init__()
        self.model       = model
        self.data_loader = data_loader
        self.criterion   = criterion
        self.adapter     = adapter

        state_forward       = use_state(lambda: model.forward)
        self._param_keys    = tuple(dict(self.model.named_parameters()).keys())
        self._model_buffers = {
            key: value 
            for key, value in state_forward.init_state().items() if key not in self._param_keys
        }

        vmap_state_forward      = vmap(state_forward, in_dims=(0, None))
        self._jit_state_forward = jit(vmap_state_forward, 
            trace=True, lazy=False, 
            example_inputs=(vmap_state_forward.init_state(), next(iter(self.data_loader))[0])
        ) 
        self._jit_forward       = torch.jit.script(model)
        self._jit_criterion     = torch.jit.script(criterion)

    @torch.jit.ignore
    def evaluate(self, pop_params: Dict[str, nn.Parameter]) -> torch.Tensor:
        if pop_params.ndim != 1 and pop_params.size(0) != 1: # vmapped evaluation
            num_map = pop_params.size(0) # FIXME
            model_buffers = {
                key: value.unsqueeze(0).expand(num_map, *([-1] * value.ndim)) 
                for key, value in self._model_buffers.items()
            }
            # state = {**model_buffers, **pop_params}
            state = model_buffers
            state.update(pop_params)
            n_inputs = 0
            for (inputs, labels) in self.data_loader:
                inputs.to(pop_params.device, non_blocking=True)
                labels.to(pop_params.device, non_blocking=True)

                state, logits = self._jit_state_forward(state, inputs)
                n_inputs     += inputs.size(0)
                result       += self._jit_criterion(logits, labels) * inputs.size(0)
    
            pop_fitness = result / n_inputs

        else: # single evaluation
            flat_params = pop_params if pop_params.ndim == 1 else pop_params[0]
            params = self.adapter.to_params(flat_params)
            self.model.load_state_dict(params)

            result = torch.tensor(0, device=pop_params.device)
            n_inputs = 0
            for (inputs, labels) in self.data_loader:
                inputs.to(pop_params.device, non_blocking=True)
                labels.to(pop_params.device, non_blocking=True)

                logits    = self._jit_forward(inputs)
                n_inputs += inputs.size(0)
                result   += self._jit_criterion(logits, labels) * inputs.size(0)
    
            pop_fitness = result / n_inputs

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

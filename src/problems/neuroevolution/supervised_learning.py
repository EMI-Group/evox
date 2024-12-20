import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable, Dict, Sequence, Tuple

from ...core import Problem, jit_class, use_state, vmap, jit
from ...core._vmap_fix import align_vmap_tensor


@jit_class
class SupervisedLearningProblem(Problem):
    def __init__(self,
        model: nn.Module,
        data_loader: DataLoader,
        criterion  : Callable,
        treeAndVec : Sequence[str] # TODO
    ):
        super().__init__()
        self.data_loader = data_loader.__iter__()
        self.model       = model
        self.criterion   = criterion

        state_forward = use_state(lambda: model.forward)
        self._param_keys = treeAndVec
        self._model_buffers = {k: v for k, v in state_forward.init_state().items() if v not in self._param_keys}
        vmap_state_forward = vmap(state_forward)
        self._jit_state_forward = jit(vmap_state_forward, trace=True, lazy=True)
        self._jit_model  = torch.jit.script(model)

    # @torch.jit.ignore
    # def _data_loader_next(self) -> Tuple[torch.Tensor, torch.Tensor]:
    #     return self.data_loader._next_data()

    # @torch.jit.ignore
    # def _data_loader_reset(self) -> None:
    #     self.data_loader._reset()

    # def _data_loader_has_next(self) -> bool:
    #     # TODO: return self.data_loader.
    #     pass

    @torch.jit.ignore
    def evaluate(self, pop_params: Dict[str, torch.nn.Parameter]) -> torch.Tensor:
        if pop_params.ndim != 1 and pop_params.size(0) != 1: # TODO: Vmapped evaluation
            num_map = pop_params.size(0) # FIXME
            model_buffers = self._model_buffers.unsqueeze(0).extend(...) # FIXME
            state = {**model_buffers, **pop_params}
            for (inputs, labels) in self.data_loader:
                state, logits = self._jit_state_forward(state, inputs)
                ...

        else: # Single evaluation
            params = pop_params if pop_params.ndim == 1 else pop_params[0]
            # TODO: params.unfold()
            self.model.load_state_dict(params)

            result = torch.tensor(0, device=pop_params.device)
            n_inputs = 0
            # for (inputs, labels) in self.data_loader:
            self._data_loader_reset()
            while self._data_loader_has_next():
                (inputs, labels) = self._data_loader_next()
                inputs.to(pop_params.device, non_blocking=True)
                labels.to(pop_params.device, non_blocking=True)

                logits = self.model(inputs)
                n_inputs += inputs.size(0)
                result += self.criterion(logits, labels) * inputs.size(0)
    
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

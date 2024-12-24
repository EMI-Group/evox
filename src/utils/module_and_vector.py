import torch
import torch.nn as nn
from typing import Dict

from ..core import jit_class, ModuleBase, jit, vmap
from ..core._vmap_fix import tree_flatten, tree_unflatten


# @jit_class
class ParamsAndVector(ModuleBase):

    def __init__(self, dummy_module: nn.Module):
        super().__init__()
        params = dict(dummy_module.named_parameters())
        flat_params, param_spec = tree_flatten(params)

        self._jit_tree_flatten   = jit(lambda x: tree_flatten(x)[0], 
            trace=True, lazy=False, example_inputs=(params,),
        )
        self._jit_tree_unflatten = jit(lambda x: tree_unflatten(x, param_spec), 
            trace=True, lazy=False, example_inputs=(flat_params,),
        )

        def prod_tuple(xs):
            prod = 1
            for x in xs:
                prod *= x
            return prod

        self.shapes        = [x.shape for x in flat_params]
        self.start_indices = []
        self.slice_sizes   = []
        index = 0
        for shape in self.shapes:
            self.start_indices.append(index)
            size = prod_tuple(shape)
            self.slice_sizes.append(size)
            index += size

    def to_vector(self, params: Dict[str, torch.nn.Parameter]) -> torch.Tensor:
        flat_params = self._jit_tree_flatten(params)
        flat_params = [x.reshape(-1) for x in flat_params] # TODO: optimization
        return torch.concat(flat_params, dim=0)

    def batched_to_vector(self, batched_params: Dict[str, torch.nn.Parameter]) -> torch.Tensor:
        flat_params = self._jit_tree_flatten(batched_params)
        flat_params = [x.reshape(x.shape[0], -1) for x in flat_params] # TODO: optimization
        return torch.concat(flat_params, dim=1)

    def to_params(self, vector: torch.Tensor) -> Dict[str, torch.nn.Parameter]:
        flat_params = []
        for start_index, slice_size, shape in zip(self.start_indices, self.slice_sizes, self.shapes):
            # TODO:
            flat_params.append(vector.narrow(dim=0, start=start_index, length=slice_size).reshape(shape))
        return self._jit_tree_unflatten(flat_params)

    # def batched_to_params(self, vectors: torch.Tensor) -> Dict[str, torch.nn.Parameter]:
    #     # TODO: vmap
    #     pass 

    # def forward(self, x: torch.Tensor) -> Dict[str, torch.nn.Parameter]:
    #     return self.batched_to_params(x)

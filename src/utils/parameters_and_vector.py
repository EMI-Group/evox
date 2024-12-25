import torch
import torch.nn as nn
from typing import Dict, List
from math import prod

from ..core import jit, jit_class, ModuleBase
from ..core._vmap_fix import tree_flatten, tree_unflatten


@jit_class
class ParamsAndVector(ModuleBase):

    def __init__(self, dummy_model: nn.Module):
        super().__init__()
        params = dict(dummy_model.named_parameters())
        flat_params, param_spec = tree_flatten(params)

        self._jit_tree_flatten = jit(lambda x: tree_flatten(x)[0], 
            trace=True, lazy=False, example_inputs=(params,),
        )
        self._jit_tree_unflatten = jit(lambda x: tree_unflatten(x, param_spec), 
            trace=True, lazy=False, example_inputs=(flat_params,),
        )

        shapes        = [x.shape for x in flat_params]
        start_indices = []
        slice_sizes   = []
        index = 0
        for shape in shapes:
            start_indices.append(index)
            size = prod(shape)
            slice_sizes.append(size)
            index += size

        self.shapes        = tuple(shapes)
        self.start_indices = tuple(start_indices)
        self.slice_sizes   = tuple(slice_sizes)

    def to_vector(self, params: Dict[str, nn.Parameter]) -> torch.Tensor:
        flat_params: List[nn.Parameter] = self._jit_tree_flatten(params)
        flat_params = [x.reshape(-1) for x in flat_params]
        return torch.concat(flat_params, dim=0)

    def batched_to_vector(self, batched_params: Dict[str, nn.Parameter]) -> torch.Tensor:
        flat_params: List[nn.Parameter] = self._jit_tree_flatten(batched_params)
        flat_params = [x.reshape(x.shape[0], -1) for x in flat_params]
        return torch.concat(flat_params, dim=1)

    def to_params(self, vector: torch.Tensor) -> Dict[str, nn.Parameter]:
        flat_params = []
        for start_index, slice_size, shape in zip(self.start_indices, self.slice_sizes, self.shapes):
            flat_params.append(
                vector.narrow(dim=0, start=start_index, length=slice_size).reshape(shape)
            )
        return self._jit_tree_unflatten(flat_params)

    def batched_to_params(self, vectors: torch.Tensor) -> Dict[str, nn.Parameter]:
        flat_params = []
        batch_size = vectors.shape[0]
        for start_index, slice_size, shape in zip(self.start_indices, self.slice_sizes, self.shapes):
            flat_params.append(
                vectors.narrow(dim=1, start=start_index, length=slice_size).reshape(batch_size, *shape)
            )
        return self._jit_tree_unflatten(flat_params)

    def forward(self, x: torch.Tensor) -> Dict[str, nn.Parameter]:
        return self.batched_to_params(x)

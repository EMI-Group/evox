from math import prod
from typing import Dict, List

import torch
import torch.nn as nn

from evox.core import ModuleBase

from .re_export import tree_flatten, tree_unflatten


class ParamsAndVector(ModuleBase):
    """The class to convert (batched) parameters dictionary to vector(s) and vice versa."""

    def __init__(self, dummy_model: nn.Module):
        """
        Initialize the ParamsAndVector instance.

        :param dummy_model: A PyTorch model whose parameters will be used to initialize the parameter and vector conversion attributes. Must be an initialized PyTorch model.
        """
        super().__init__()
        params = dict(dummy_model.named_parameters())
        flat_params, self.param_spec = tree_flatten(params)

        shapes = [x.shape for x in flat_params]
        start_indices = []
        slice_sizes = []
        index = 0
        for shape in shapes:
            start_indices.append(index)
            size = prod(shape)
            slice_sizes.append(size)
            index += size

        self.shapes = tuple(shapes)
        self.start_indices = tuple(start_indices)
        self.slice_sizes = tuple(slice_sizes)

    def _tree_flatten(self, x: Dict[str, nn.Parameter]) -> List[nn.Parameter]:
        return tree_flatten(x)[0]

    def _tree_unflatten(self, x: List[nn.Parameter]) -> Dict[str, nn.Parameter]:
        return tree_unflatten(x, self.param_spec)

    def to_vector(self, params: Dict[str, nn.Parameter]) -> torch.Tensor:
        """Convert the input parameters dictionary to a single vector.

        :param params: The input parameters dictionary.

        :return: The output vector obtained by concatenating the flattened parameters.
        """

        flat_params: List[nn.Parameter] = self._tree_flatten(params)
        flat_params = [x.reshape(-1) for x in flat_params]
        return torch.concat(flat_params, dim=0)

    def batched_to_vector(self, batched_params: Dict[str, nn.Parameter]) -> torch.Tensor:
        """Convert a batched parameters dictionary to a batch of vectors.

        The input dictionary values must be batched parameters, i.e., they must have the same shape at the first dimension.

        :param batched_params: The input batched parameters dictionary.

        :return: The output vectors obtained by concatenating the flattened batched parameters. The first dimension of the output vector corresponds to the batch size.
        """
        flat_params: List[nn.Parameter] = self._tree_flatten(batched_params)
        flat_params = [x.reshape((x.size(0), -1)) for x in flat_params]
        return torch.concat(flat_params, dim=1)

    def to_params(self, vector: torch.Tensor) -> Dict[str, nn.Parameter]:
        """Convert a vector back to a parameters dictionary.

        :param vector: The input vector representing flattened model parameters.

        :return: The reconstructed parameters dictionary.
        """
        flat_params = []
        for start_index, slice_size, shape in zip(self.start_indices, self.slice_sizes, self.shapes):
            flat_params.append(vector.narrow(dim=0, start=start_index, length=slice_size).reshape(shape))
        return self._tree_unflatten(flat_params)

    def batched_to_params(self, vectors: torch.Tensor) -> Dict[str, nn.Parameter]:
        """Convert a batch of vectors back to a batched parameters dictionary.

        :param vectors: The input batch of vectors representing flattened model parameters. The first dimension of the tensor corresponds to the batch size.

        :return: The reconstructed batched parameters dictionary whose tensors' first dimensions correspond to the batch size.
        """
        flat_params = []
        batch_size = vectors.size(0)
        for start_index, slice_size, shape in zip(self.start_indices, self.slice_sizes, self.shapes):
            flat_params.append(vectors.narrow(dim=1, start=start_index, length=slice_size).reshape(batch_size, *shape))
        return self._tree_unflatten(flat_params)

    def forward(self, x: torch.Tensor) -> Dict[str, nn.Parameter]:
        """The forward function for the `ParamsAndVector` module is an alias of `batched_to_params` to cope with `StdWorkflow`."""
        return self.batched_to_params(x)

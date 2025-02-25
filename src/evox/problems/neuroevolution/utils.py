import copy
from typing import Callable, Dict, NamedTuple, Tuple

import torch
import torch.nn as nn

from evox.core import use_state, vmap


class ModelStateForwardResult(NamedTuple):
    init_state: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
    state_forward: Callable


def get_vmap_model_state_forward(
    model: nn.Module,
    pop_size: int,
    device: torch.device,
    in_dims=(0, None),
    randomness="different",
) -> ModelStateForwardResult:
    """Get model state forward function for vmap and non-vmap models.
    When `get_non_vmap` is False, the function returns only vmap model state forward function.
    When `get_non_vmap` is True, the function returns both vmap and non-vmap model state forward functions.
    """
    # Model initialization
    inference_model = copy.deepcopy(model)
    inference_model = inference_model.to(device=device)
    state_forward = use_state(inference_model)
    vmap_state_forward = torch.compile(vmap(state_forward, in_dims=in_dims, randomness=randomness))
    params, buffers = torch.func.stack_module_state([inference_model] * pop_size)
    vmap_model_init_state = params | buffers

    for _, value in vmap_model_init_state.items():
        value.requires_grad = False

    return ModelStateForwardResult(vmap_model_init_state, vmap_state_forward)

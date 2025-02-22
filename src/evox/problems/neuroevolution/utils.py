import copy
import types
from typing import Any, Callable, Dict, NamedTuple, Tuple

import torch
import torch.nn as nn

from ...core import use_state, vmap
from ...core.module import assign_load_state_dict


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
    for _, value in inference_model.named_parameters():
        value.requires_grad = False

    state_forward = use_state(inference_model)
    vmap_state_forward = torch.compile(vmap(state_forward, in_dims=in_dims, randomness=randomness))
    vmap_model_init_state = torch.func.stack_module_state([inference_model] * pop_size)
    return ModelStateForwardResult(vmap_model_init_state, vmap_state_forward)

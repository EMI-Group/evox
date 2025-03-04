import copy
import types
from typing import Any, Callable, Dict, NamedTuple, Tuple

import torch
import torch.nn as nn

from ...core import jit, use_state, vmap
from ...core.module import assign_load_state_dict


class ModelStateForwardResult(NamedTuple):
    init_state: Dict[str, torch.Tensor]
    state_forward: Callable
    dummy_output: Any
    param_to_state_key_map: Dict[str, str]
    model_buffers: Dict[str, torch.Tensor]


def get_vmap_model_state_forward(
    model: nn.Module,
    pop_size: int,
    dummy_inputs: torch.Tensor,
    check_output: Callable[[Any], bool],
    device: torch.device,
    vmap_in_dims=(0, None),
    get_non_vmap: bool = False,
    dummy_single_inputs: torch.Tensor | None = None,
    check_single_output: Callable[[Any], bool] | None = None,
) -> ModelStateForwardResult | Tuple[ModelStateForwardResult, ModelStateForwardResult]:
    """Get model state forward function for vmap and non-vmap models.
    When `get_non_vmap` is False, the function returns only vmap model state forward function.
    When `get_non_vmap` is True, the function returns both vmap and non-vmap model state forward functions.
    """
    # Model initialization
    inference_model = copy.deepcopy(model)
    inference_model = inference_model.to(device=device)
    for _, value in inference_model.named_parameters():
        value.requires_grad = False
    inference_model.load_state_dict = types.MethodType(assign_load_state_dict, inference_model)

    # JITed and vmapped model state forward initialization
    state_forward = use_state(lambda: inference_model.forward)
    model_init_state = state_forward.init_state(clone=False)
    if get_non_vmap:
        if dummy_single_inputs is None:
            dummy_single_inputs = dummy_inputs
        jit_state_forward, (_, dummy_single_outputs) = jit(
            state_forward,
            trace=True,
            lazy=False,
            example_inputs=(model_init_state, dummy_single_inputs),
            return_dummy_output=True,
        )
        assert check_single_output(
            dummy_single_outputs
        ), f"Single forward evaluation assertion failed for {dummy_single_outputs}"

    vmap_state_forward = vmap(state_forward, in_dims=vmap_in_dims)
    vmap_model_init_state = vmap_state_forward.init_state(pop_size, expand=False)
    jit_vmap_state_forward, (_, dummy_vmap_outputs) = jit(
        vmap_state_forward,
        trace=True,
        lazy=False,
        example_inputs=(vmap_model_init_state, dummy_inputs),
        return_dummy_output=True,
    )
    assert check_output(dummy_vmap_outputs), f"Mapped forward evaluation assertion failed for {dummy_vmap_outputs}"

    # Building map from model parameters key to model state key
    model_params = dict(inference_model.named_parameters())
    param_to_state_key_map: Dict[str, str] = {
        params_key: state_key
        for state_key, state_value in model_init_state.items()
        for params_key, params_value in model_params.items()
        if torch.equal(state_value, params_value)
    }
    # Model parameters and buffers registration
    if get_non_vmap:
        model_buffers = {key: value for key, value in model_init_state.items() if key not in param_to_state_key_map}
        non_vmap_result = ModelStateForwardResult(
            model_init_state,
            jit_state_forward,
            dummy_single_outputs,
            param_to_state_key_map,
            model_buffers,
        )

    vmap_model_buffers = {key: value for key, value in vmap_model_init_state.items() if key not in param_to_state_key_map}
    vmap_result = ModelStateForwardResult(
        vmap_model_init_state,
        jit_vmap_state_forward,
        dummy_vmap_outputs,
        param_to_state_key_map,
        vmap_model_buffers,
    )

    if get_non_vmap:
        return non_vmap_result, vmap_result
    return vmap_result

import copy
import types
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Iterable, Iterator

from ...core import Problem, jit_class, use_state, vmap, jit
from ...core.module import assign_load_state_dict


__supervised_data__: Dict[int, Dict[str, DataLoader | Iterable | Iterator | Tuple]] = None


@jit_class
class SupervisedLearningProblem(Problem):

    def __init__(self, 
        model      : nn.Module,
        data_loader: DataLoader, 
        criterion  : nn.Module,
        pop_size   : int,
        device     : torch.device | None = None,
    ):
        super().__init__()
        # Register global data loader
        global __supervised_data__
        if __supervised_data__ is None:
            __supervised_data__ = {}
        instance_id = hash(self)
        if instance_id not in __supervised_data__.keys():
            __supervised_data__[instance_id] = {
                "data_loader_ref" : data_loader,
                "data_loader_iter": None,
                "data_next_cache" : None,
            }
    
        device = torch.get_default_device() if device is None else device
        inference_model = copy.deepcopy(model) 
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
        self._sample_param_ndim_tuple = (
            param_keys[0], 
            inference_model.get_parameter(param_keys[0]).ndim
        )

        try:
            dummy_inputs, dummy_labels = next(iter(data_loader))
        except StopIteration:
            raise RuntimeError(
                f"The `data_loader` of `{self.__class__.__name__}` must contain at least one item."
            )
        dummy_inputs = dummy_inputs.to(device=device)
        dummy_labels = dummy_labels.to(device=device)

        vmap_state_forward = vmap(state_forward, in_dims=(0, None))
        model_init_state   = vmap_state_forward.init_state(pop_size)
        self._jit_vmap_state_forward, (_, dummy_logits) = jit(vmap_state_forward, 
            trace=True, lazy=False, return_dummy_output=True, 
            example_inputs=(model_init_state, dummy_inputs)
        )
        self._jit_forward  = torch.jit.script(inference_model)

        state_criterion    = use_state(lambda: criterion.forward)
        if len(state_criterion.init_state()) > 0:
            vmap_state_criterion = vmap(state_criterion, in_dims=(0, 0, None))
            criterion_init_state = vmap_state_criterion.init_state(pop_size)
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
        global __supervised_data__
        # TODO: change to while-loop
        for v in __supervised_data__[self._hash_id_]["data_loader_ref"]: # get data loader by inner hash ID
            inputs: torch.Tensor = v[0]
            labels: torch.Tensor = v[1]
            inputs = inputs.to(device=device, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)

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
        global __supervised_data__
        for (inputs, labels) in __supervised_data__[self._hash_id_]:
            inputs.to(device, non_blocking=True)
            labels.to(device, non_blocking=True)

            logits    = self._jit_forward(inputs)
            n_inputs += inputs.size(0)
            result   += self._jit_criterion(logits, labels) * inputs.size(0)

        pop_fitness = result / n_inputs
        return pop_fitness

    def evaluate(self, pop_params: Dict[str, nn.Parameter]) -> torch.Tensor:
        pop_params_value = pop_params[self._sample_param_ndim_tuple[0]]
        if pop_params_value.ndim > self._sample_param_ndim_tuple[1] and pop_params_value.size(0) != 1:
            # vmapped evaluation
            pop_fitness = self._evaluate_vmap(pop_params, pop_params_value.size(0), pop_params_value.device)
        else: 
            # single evaluation
            pop_fitness = self._evaluate_single(pop_params, pop_params_value.device)
        return pop_fitness 

    ###############
    @torch.jit.ignore
    def _data_loader_reset(self) -> None:
        global __supervised_data__
        data_info = __supervised_data__[self._hash_id_]
        data_info["data_loader_iter"] = iter(data_info["data_loader_ref"])
        try:
            data_info["data_next_cache"] = next(data_info["data_loader_iter"])
        except StopIteration:
            data_info["data_next_cache"] = None

    @torch.jit.ignore
    def _data_loader_next(self) -> Tuple[torch.Tensor, torch.Tensor]:
        global __supervised_data__
        data_info = __supervised_data__[self._hash_id_]
        next_data = data_info["data_next_cache"]
        try:
            data_info["data_next_cache"] = next(data_info["data_loader_iter"])
        except StopIteration:
            data_info["data_next_cache"] = None
        return next_data

    @torch.jit.ignore
    def _data_loader_has_next(self) -> bool:
        global __supervised_data__
        return __supervised_data__[self._hash_id_]["data_next_cache"] is not None

    # def evaluate(self, pop_params: Dict[str, nn.Parameter]) -> torch.Tensor:
    #     self._data_loader_reset()
    #     while self._data_loader_has_next():
    #         inputs, labels = self._data_loader_next()
    #         print(inputs.shape, inputs.dtype, inputs.sum())
    #         print(labels.shape, labels.dtype, labels.sum())
    #     return torch.tensor([1,])

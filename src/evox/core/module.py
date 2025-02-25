import copy
from functools import wraps
from typing import (
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import torch
import torch.nn as nn
from torch.overrides import TorchFunctionMode


def _if_none(a, b):
    return b if a is None else a


def _is_magic(name: str):
    return name.startswith("__") and name.endswith("__")


ParameterT = TypeVar("ParameterT", torch.Tensor, int, float, complex)


def Parameter(
    value: ParameterT,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> ParameterT:
    """Wraps a value as parameter with `requires_grad=False`.

    :param value: The parameter value.
    :param dtype: The dtype of the parameter. Defaults to None.
    :param device: The device of the parameter. Defaults to None.
    :param requires_grad: Whether the parameter requires gradient. Defaults to False.

    :return: The parameter.
    """
    return nn.Parameter(
        (
            value.to(dtype=dtype, device=device)
            if isinstance(value, torch.Tensor)
            else torch.as_tensor(value, dtype=dtype, device=device)
        ),
        requires_grad=requires_grad,
    )


def Mutable(value: torch.Tensor, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Wraps a value as a mutable tensor.

    :param value: The value to be wrapped.
    :param dtype: The dtype of the tensor. Defaults to None.
    :param device: The device of the tensor. Defaults to None.

    :return: The wrapped tensor.
    """
    return nn.Buffer(value.to(dtype=dtype, device=device))


def assign_load_state_dict(self: nn.Module, state_dict: Mapping[str, torch.Tensor]):
    """Copy parameters and buffers from state_dict into this module and its descendants.

    This method is used to mimic the behavior of `ModuleBase.load_state_dict` so that a regular `nn.Module` can be used with `vmap`.

    ## Usage:
    ```
    import types
    # ...
    model = ... # define your model
    model.load_state_dict = types.MethodType(assign_load_state_dict, model)
    vmap_forward = vmap(use_state(model.forward))
    jit_forward = jit(vmap_forward, trace=True, example_inputs=(vmap_forward.init_state(), ...)) # JIT trace forward pass of the model
    ```
    """
    assert not isinstance(self, ModuleBase), "Cannot call `assign_load_state_dict` on `ModuleBase`"
    sub_modules: Dict[str, Dict[str, torch.Tensor]] = {}
    for k, v in state_dict.items():
        if "." in k:
            sub_key, sub_mod = ((t := k.split(".", 1))[1], t[0])
            if sub_mod not in sub_modules:
                sub_modules[sub_mod] = {}
            sub_modules[sub_mod][sub_key] = v
        else:
            setattr(self, k, v)
    if len(sub_modules) > 0:
        for k, v in sub_modules.items():
            assign_load_state_dict(getattr(self, k), v)


class ModuleBase(nn.Module):
    """
    The base module for all algorithms and problems in the library.

    ## Notice
    1. This module is an object-oriented one that can contain mutable values.
    2. Functional programming model is supported via `self.state_dict(...)` and `self.load_state_dict(...)`.
    3. The module initialization for non-static members are recommended to be written in the overwritten method of `setup` (or any other member method) rather than `__init__`.
    4. Basically, predefined submodule(s) which will be ADDED to this module and accessed later in member method(s) should be treated as "non-static members", while any other member(s) should be treated as "static members".

    ## Usage
    1. Static methods to be JIT shall be defined as is, e.g.,
    ```
    @jit
    def func(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass
    ```
    2. If a class member function with python dynamic control flows like `if` were to be JIT, a separated static method with `jit(..., trace=False)` or `torch.jit.script_if_tracing` shall be used:
    ```
    class ExampleModule(ModuleBase):
        def setup(self, mut: torch.Tensor):
            self.add_mutable("mut", mut)
            # or
            self.mut = Mutable(mut)
            return self

        @partial(jit, trace=False)
        def static_func(x: torch.Tensor, threshold: float) -> torch.Tensor:
            if x.flatten()[0] > threshold:
                return torch.sin(x)
            else:
                return torch.tan(x)
        @jit
        def jit_func(self, p: torch.Tensor) -> torch.Tensor:
            x = ExampleModule.static_func(p, self.threshold)
            ...
    ```
    3. `ModuleBase` is usually used with `jit_class` to automatically JIT all non-magic member methods:
    ```

    class ExampleModule(ModuleBase):
        # This function will be automatically JIT
        def func1(self, x: torch.Tensor) -> torch.Tensor:
            pass

        # Use `torch.jit.ignore` to disable JIT and leave this function as Python callback
        @torch.jit.ignore
        def func2(self, x: torch.Tensor) -> torch.Tensor:
            # you can implement pure Python logic here
            pass

        # JIT functions can invoke other JIT functions as well as non-JIT functions
        def func3(self, x: torch.Tensor) -> torch.Tensor:
            y = self.func1(x)
            z = self.func2(x)
            pass
    ```
    """

    def __init__(self, *args, **kwargs):
        """Initialize the ModuleBase.

        :param *args: Variable length argument list, passed to the parent class initializer.
        :param **kwargs: Arbitrary keyword arguments, passed to the parent class initializer.

        Attributes:
            __static_names__ (list): A list to store static member names.
        """

        super().__init__(*args, **kwargs)
        self.train(False)

    def eval(self):
        assert False, "`ModuleBase.eval()` shall never be invoked to prevent ambiguity."

    def setup(self, *args, **kwargs):
        """Setup the module.
        Module initialization lines should be written in the overwritten method of `setup` rather than `__init__`.

        :return: This module.

        ## Notice
        The static initialization can still be written in the `__init__` while the mutable initialization cannot.
        Therefore, multiple calls of `setup` for multiple initializations are possible.
        """
        return self

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor], copy: bool = False, **kwargs):
        """Copy parameters and buffers from state_dict into this module and its descendants.
        Overwrites [`torch.nn.Module.load_state_dict`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict).

        :param state_dict: A dict containing parameters and buffers used to update this module. See `torch.nn.Module.load_state_dict`.
        :param copy: Use the original `torch.nn.Module.load_state_dict` to copy the `state_dict` to current state (`copy=True`) or use this implementation that assigns the values of this module to the ones in the `state_dict` (`copy=False`). Defaults to False.
        :param **kwargs: The original arguments of `torch.nn.Module.load_state_dict`. Ignored if `copy=False`.

        :return: If `copy=True`, returns the return of `torch.nn.Module.load_state_dict`; otherwise, no return.
        """
        if copy:
            return super().load_state_dict(state_dict, **kwargs)
        # else
        sub_modules: Dict[str, Dict[str, torch.Tensor]] = {}
        for k, v in state_dict.items():
            if "." in k:
                sub_key, sub_mod = ((t := k.split(".", 1))[1], t[0])
                if sub_mod not in sub_modules:
                    sub_modules[sub_mod] = {}
                sub_modules[sub_mod][sub_key] = v
            else:
                if isinstance(self.__getattr_inner__(k), nn.Parameter) and not isinstance(v, nn.Parameter):
                    v = nn.Parameter(v, requires_grad=v.requires_grad)
                self.__setattr_inner__(k, v)
        if len(sub_modules) > 0:
            for k, v in sub_modules.items():
                getattr(self, k).load_state_dict(v)

    def add_mutable(
        self,
        name: str,
        value: Union[
            torch.Tensor | nn.Module,
            Sequence[torch.Tensor | nn.Module],
            Dict[str, torch.Tensor | nn.Module],
        ],
    ) -> None:
        """Define a mutable value in this module that can be accessed via `self.[name]` and modified in-place.

        :param name: The mutable value's name.
        :param value: The mutable value, can be a tuple, list, dictionary of a `torch.Tensor`.

        :raises NotImplementedError: If the mutable value's type is not supported yet.
        :raises AssertionError: If the `name` is invalid.
        """
        assert name.isdigit() or str.isidentifier(name), f"Name {name} is not a valid Python name."
        if isinstance(value, torch.Tensor):
            setattr(self, name, nn.Buffer(value))
        elif isinstance(value, nn.Module):
            super().__setattr__(name, value)
        elif isinstance(value, tuple) or isinstance(value, list):
            all_tensors = all(map(lambda x: isinstance(x, torch.Tensor), value))
            all_modules = all(map(lambda x: isinstance(x, nn.Module), value))
            assert all_tensors or all_modules, (
                "Expect a tuple or list of `torch.Tensor` or `nn.Module`, got a mixture of both or none of them."
            )
            if all_modules:
                sub_module = nn.ModuleList(value)
                self.add_module(name, sub_module)
            else:
                sub_module = ModuleBase()
                for i, v in enumerate(value):
                    sub_module.add_mutable(str(i), v)
                self.add_module(name, sub_module)
        elif isinstance(value, dict):
            all_tensors = all(map(lambda x: isinstance(x, torch.Tensor), value.values()))
            all_modules = all(map(lambda x: isinstance(x, nn.Module), value.values()))
            assert all_tensors or all_modules, (
                "Expect a dict of `torch.Tensor` or `nn.Module`, got a mixture of both or none of them."
            )
            if all_modules:
                sub_module = nn.ModuleDict(value)
                self.add_module(name, sub_module)
            else:
                sub_module = ModuleBase()
                for k, v in value.items():
                    sub_module.add_mutable(k, v)
                self.add_module(name, sub_module)
        else:
            raise NotImplementedError(f"Mutable of type {type(value)} is not supported yet.")

    def __getattr_inner__(self, name):
        try:
            value = super(nn.Module, self).__getattribute__(name)
        except Exception:
            value = super(ModuleBase, self).__getattr__(name)
        return value

    def __delattr_inner__(self, name):
        try:
            object.__delattr__(self, name)
        except Exception:
            super(ModuleBase, self).__delattr__(name)

    def __setattr_inner__(self, name, value):
        super().__setattr__(name, value)

    def __getitem__(self, key: Union[int, slice, str]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Get the mutable value(s) stored in this list-like module.

        :param key: The key used to index mutable value(s).

        :raises IndexError: If `key` is out of range.
        :raises TypeError: If `key` is of wrong type.

        :return: The indexed mutable value(s).
        """
        if isinstance(key, slice):
            key = range(_if_none(key.start, 0), key.stop, _if_none(key.step, 1))
            return [self.__getitem__(k) for k in key]
        elif isinstance(key, int):
            if key < 0:
                raise IndexError(f"The index {key} cannot be negative.")
            key = str(key)
        if isinstance(key, str) and key in self._buffers:
            return self.get_buffer(key)
        else:
            raise TypeError(f"Invalid argument type {type(key)}.")

    def __setitem__(
        self,
        value: Union[torch.Tensor, List[torch.Tensor]],
        key: Union[slice, int],
    ) -> None:
        """Set the mutable value(s) stored in this list-like module.

        :param value: The new mutable value(s).
        :param key: The key used to index mutable value(s).
        """
        targets = self.__getitem__(key)
        if isinstance(targets, list):
            value = list(value)
            assert len(value) == len(targets), "Length of value mismatch, expected {len(targets)}, got {len(value)}"
            for t, v in zip(targets, value):
                t.set_(v)
        else:
            assert isinstance(value, torch.Tensor), f"Type of value mismatch, expected torch.Tensor, got {type(value)}"
            targets.set_(value)

    def iter(self) -> Tuple[torch.Tensor]:
        if len(self._buffers) > 0:
            return tuple(self.buffers(recurse=False))
        else:
            return tuple(self.modules())


# We still need a fix for the vmap
# related issue: https://github.com/pytorch/pytorch/issues/124423
class TransformGetSetItemToIndex(TorchFunctionMode):
    # This is needed since we want to support calling
    # A[idx] or A[idx] += b, where idx is a scalar tensor.
    # When idx is a scalar tensor, Torch implicitly convert it to a python
    # scalar and create a view of A.
    # Workaround: We convert the scalar tensor to a 1D tensor with one element.
    # That is, we convert A[idx] to A[idx[None]][0], A[idx] += 1 to A[idx[None]] += 1.
    # This is a temporary solution until the issue is fixed in PyTorch.
    def __torch_function__(self, func, types, args, kwargs=None):
        if func == torch.Tensor.__getitem__:
            x, index = args
            if isinstance(index, torch.Tensor) and index.ndim == 0:
                return func(x, index[None], **(kwargs or {}))[0]
                # return torch.index_select(x, 0, index)
        elif func == torch.Tensor.__setitem__:
            x, index, value = args
            if isinstance(index, torch.Tensor) and index.ndim == 0:
                return func(x, index[None], value, **(kwargs or {}))

        return func(*args, **(kwargs or {}))


@wraps(torch.vmap)
def vmap(*args, **kwargs) -> Callable:
    """Fix the torch.vmap's issue with __getitem__ and __setitem__.
    Related issue: https://github.com/pytorch/pytorch/issues/124423.
    """

    vmapped = torch.vmap(*args, **kwargs)

    def wrapper(*args, **kwargs):
        with TransformGetSetItemToIndex():
            return vmapped(*args, **kwargs)

    return wrapper


def use_state(stateful_func: Union[Callable, nn.Module]) -> Callable:
    """Transform a nn.module's method or an nn.module into a stateful function.
    When using nn.module, the stateful version of the default `forward` method will be created.
    The stateful function will have a signature of fn(params_and_buffers, *args, **kwargs) -> params_and_buffers | Tuple[params_and_buffers, Any]].

    ## Examples

    ```python
    from evox import use_state, vmap
    workflow = ... # define your workflow
    stateful_step = use_state(workflow.step)
    vmap_stateful_step = vmap(stateful_step)
    batch_state = torch.func.stack_module_states([workflow] * 3)
    new_batch_state = vmap_stateful_step(batch_state)
    ```
    """
    if not isinstance(stateful_func, torch.nn.Module):
        original_module = stateful_func.__self__
        module = copy.copy(original_module)
        module.forward = stateful_func
    else:
        module = stateful_func

    def wrapper(params_and_buffers, *args, **kwargs):
        output = torch.func.functional_call(module, params_and_buffers, *args, **kwargs)
        if output is None:
            return params_and_buffers
        else:
            return params_and_buffers, output

    return wrapper

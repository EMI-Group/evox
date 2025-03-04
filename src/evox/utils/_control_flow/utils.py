import inspect
from typing import Callable, Protocol, Tuple, TypeVar

import torch


def _get_cache_key_object(cache_dict: dict, *fns: Callable):
    fn_ids = []
    for fn in fns:
        fn_id = getattr(fn, "__id__", id(fn))
        if inspect.isfunction(fn) and fn.__name__ == "<lambda>":
            fn_id = fn.__code__.co_code
        fn_ids.append(fn_id)
    key = tuple(fn_ids)
    if key in cache_dict:
        return cache_dict[key]
    else:
        return key


def _param_clone(v: torch.Tensor):
    if isinstance(v, torch.nn.Parameter):
        return torch.nn.Parameter(v.clone(), requires_grad=v.requires_grad)
    else:
        return v.clone()


T = TypeVar("T", bound=torch.Tensor)
R = TypeVar("R", bound=torch.Tensor)


class VarArgsCallable(Protocol[T, R]):
    def __call__(self, *args: T) -> R:
        pass


class VarArgsCallableMultiRet(Protocol[T, R]):
    def __call__(self, *args: T) -> Tuple[R, ...]:
        pass

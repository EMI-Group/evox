from jax.tree_util import register_pytree_node
import dataclasses
from typing import Annotated, Any, Callable, Optional, Tuple, TypeVar, get_type_hints

from typing_extensions import (
    dataclass_transform,  # pytype: disable=not-supported-yet
)

from .distributed import ShardingType


def pytree_field(
    *, static=False, stack=False, sharding=ShardingType.REPLICATED, **kwargs
):
    """ """
    metadata = {"static": static, "stack": stack, "sharding": sharding}
    kwargs.setdefault("metadata", {}).update(metadata)

    return dataclasses.field(**kwargs)


def dataclass(cls, *args, **kwargs):
    """
    A dataclass decorator that also registers the dataclass as a pytree node.
    """
    cls = dataclasses.dataclass(cls, *args, **kwargs)

    field_info = []
    # normal dataclass fields
    for field in dataclasses.fields(cls):
        is_static = field.metadata.get("static", False)
        field_info.append((field.name, field.init, is_static))
    # evox Stateful fields
    field_info.append(("_node_id", False, True))
    field_info.append(("_module_name", False, True))

    def flatten(dataclass_obj):
        children = []
        aux_data = []
        for field_name, _, is_static in field_info:
            if hasattr(dataclass_obj, field_name):
                value = getattr(dataclass_obj, field_name)
            else:
                value = None

            if is_static:
                aux_data.append(value)
            else:
                children.append(value)

        return (children, aux_data)

    def unflatten(aux_data, children):
        init_params = {}
        non_init_params = {}
        iter_aux = iter(aux_data)
        iter_children = iter(children)
        for field_name, is_init, is_static in field_info:
            if is_init:
                if is_static:
                    init_params[field_name] = next(iter_aux)
                else:
                    init_params[field_name] = next(iter_children)
            else:
                if is_static:
                    non_init_params[field_name] = next(iter_aux)
                else:
                    non_init_params[field_name] = next(iter_children)

        obj = object.__new__(cls)
        for key, value in init_params.items():
            object.__setattr__(obj, key, value)
        for key, value in non_init_params.items():
            object.__setattr__(obj, key, value)

        return obj

    register_pytree_node(cls, flatten, unflatten)

    # Add a method to set frozen attributes after init
    cls.set_frozen_attr = lambda self, key, value: object.__setattr__(self, key, value)
    return cls


@dataclass_transform(field_specifiers=(pytree_field,), kw_only_default=True)
class PyTreeNode:
    def __init_subclass__(cls, **kwargs):
        dataclass(cls, **kwargs)

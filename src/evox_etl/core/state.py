"""State helpers: pytree-agnostic `replace`, nested get/set, and etl tree re-exports."""

from __future__ import annotations

import dataclasses
from typing import Any, Iterable, TypeVar

from etl import tree_flatten, tree_leaves, tree_map, tree_unflatten

__all__ = [
    "replace",
    "get_nested",
    "set_nested",
    "tree_map",
    "tree_leaves",
    "tree_flatten",
    "tree_unflatten",
]

T = TypeVar("T")


def replace(obj: T, **changes: Any) -> T:
    """Return a new object of the same type with some fields replaced.

    Works for frozen dataclasses (via `dataclasses.replace`) and namedtuples
    (via `_replace`).
    """
    if hasattr(type(obj), "_replace"):
        return type(obj)._replace(obj, **changes)
    return dataclasses.replace(obj, **changes)


def _to_tuple(path: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(path, str):
        return tuple(path.split("."))
    return tuple(path)


def get_nested(obj: Any, path: str | Iterable[str]) -> Any:
    """Fetch a nested attribute via a dotted path ("a.b.c") or a tuple of names."""
    for key in _to_tuple(path):
        obj = getattr(obj, key)
    return obj


def set_nested(obj: T, path: str | Iterable[str], value: Any) -> T:
    """Return a copy of `obj` with the nested attribute at `path` set to `value`.

    The chain is rebuilt from the innermost object outward using `replace`.
    """
    keys = _to_tuple(path)
    if not keys:
        raise ValueError("path must be a non-empty dotted string or a sequence of attribute names")
    *parents, leaf = keys
    updated = replace(get_nested(obj, parents), **{leaf: value})
    for i in range(len(parents) - 1, -1, -1):
        updated = replace(get_nested(obj, parents[:i]), **{parents[i]: updated})
    return updated

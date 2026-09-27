"""Extension autoloading for the ETL (functional) EvoX rewrite.

Mirrors :mod:`evox_ext.autoload_ext` for :mod:`evox_etl`. Extensions are
discovered through the PEP 420 namespace package ``evox_etl_ext`` -- one
directory per domain (``utils``, ``algorithms``, ``problems``, ``operators``,
``metrics``) holding plain ``.py`` modules and NO ``__init__.py``::

    evox_etl_ext/
        algorithms/
            my_ga.py     # plain functions + frozen config dataclasses + make_*

ETL extension contract: the functional model has NO ``ModuleBase`` classes, so
an extension module exposes exactly what a built-in ``evox_etl`` module does --
plain module-level functions, frozen (config) dataclasses and ``make_*``
constructors (see ``src/evox_etl/DESIGN.md``).

``auto_load_extensions()`` discovers every installed extension and merges it
into the matching ``evox_etl.<domain>`` module. It is guarded (a missing
extension namespace is skipped silently) and fully idempotent (calling it twice
duplicates no ``__all__`` entry and re-attaches nothing).
"""

import importlib
import inspect
import pkgutil
import types
from collections.abc import Iterator
from typing import Any

# Extension namespaces and their target ``evox_etl`` modules, in order.
DOMAINS: tuple[str, ...] = ("utils", "algorithms", "problems", "operators", "metrics")


def iter_namespace(ns_pkg: types.ModuleType) -> Iterator[tuple[Any, str, bool]]:
    """Yield ``(finder, absolute_name, ispkg)`` for each submodule of ``ns_pkg``."""
    # The prefix makes each returned name absolute, so importlib can import it
    # directly without any further name mangling.
    return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")


def _extend_all(module: types.ModuleType, name: str) -> None:
    """Append ``name`` to ``module.__all__`` exactly once, creating it if absent."""
    names = getattr(module, "__all__", None)
    if names is None:
        module.__all__ = [name]
    elif name not in names:
        names.append(name)


def load_extension(ext_pkg: types.ModuleType, target_module: types.ModuleType) -> None:
    """Recursively merge extension namespace ``ext_pkg`` into ``target_module``.

    Each extension submodule is attached on the target under its leaf name. When
    a leaf already names a module on the target, the extension is merged into it
    (its top-level functions/classes are lifted) instead of clobbering it.
    Top-level functions and classes of ``ext_pkg`` itself are lifted as well.
    Idempotent: a second call re-attaches nothing and adds no ``__all__`` dupes.
    """
    if hasattr(ext_pkg, "__path__"):
        for _finder, name, _ispkg in iter_namespace(ext_pkg):
            ext_module = importlib.import_module(name)
            leaf = name.rpartition(".")[2]
            if leaf in target_module.__dict__:
                existing = target_module.__dict__[leaf]
                # Merge into an already-present submodule; a non-module attr wins.
                if isinstance(existing, types.ModuleType) and existing is not ext_module:
                    load_extension(ext_module, existing)
            else:
                setattr(target_module, leaf, ext_module)
                _extend_all(target_module, leaf)

    for attr_name in dir(ext_pkg):
        attr = getattr(ext_pkg, attr_name)
        if inspect.isfunction(attr) or inspect.isclass(attr):
            setattr(target_module, attr_name, attr)
            _extend_all(target_module, attr_name)


def auto_load_extensions() -> None:
    """Discover and merge every installed ``evox_etl_ext`` extension.

    Each domain is guarded: if the extension namespace (or the target module) is
    absent it is skipped silently, so this is a clean no-op when nothing is
    installed. Safe to call more than once.
    """
    for domain in DOMAINS:
        try:
            target_module = importlib.import_module(f"evox_etl.{domain}")
            ext_pkg = importlib.import_module(f"evox_etl_ext.{domain}")
        except ImportError:
            continue
        load_extension(ext_pkg, target_module)

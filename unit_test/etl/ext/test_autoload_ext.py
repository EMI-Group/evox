"""Tests for ``evox_etl_ext.autoload_ext`` (ETL only, NO torch).

Covers the three required behaviours of the extension autoloader:

* clean NO-OP when no extension is installed (and still a no-op when called
  twice);
* DISCOVERY + MERGE of a synthetic extension placed on a temporary ``sys.path``
  entry -- the module lands on ``evox_etl.<domain>`` and its function/class join
  the target's ``__all__``;
* ``auto_load_extensions()`` is IDEMPOTENT (a second call adds no duplicate).

A synthetic ``evox_etl_ext`` namespace is built under ``tmp_path`` and appended
to ``sys.path``; the fixture snapshots the ``evox_etl`` domain modules and
restores them (plus ``sys.path`` and ``sys.modules``) on teardown so nothing
leaks into the other collected tests.
"""

import dataclasses
import importlib
import sys
import textwrap
import types
from collections.abc import Callable

import pytest

from evox_etl_ext.autoload_ext import auto_load_extensions

DOMAINS = ("utils", "algorithms", "problems", "operators", "metrics")
DOMAIN = "operators"
MOD_NAME = "probe_ext"
QUALIFIED = f"evox_etl_ext.{DOMAIN}.{MOD_NAME}"
COLLIDING = "sampling"  # pre-existing built-in submodule of evox_etl.operators

# A synthetic extension module in the ETL functional style: plain functions +
# a frozen config dataclass + a ``make_*`` constructor (no ModuleBase classes).
EXT_SOURCE = textwrap.dedent(
    '''
    """Synthetic evox_etl extension module (functional model)."""

    import dataclasses


    @dataclasses.dataclass(frozen=True)
    class ProbeConfig:
        """Frozen config dataclass, ETL extension style."""

        value: int = 1


    def make_probe(value=1):
        """Construct a frozen :class:`ProbeConfig` (mirrors the ``make_*`` contract)."""
        return ProbeConfig(value=value)
    '''
)

# A synthetic extension whose leaf name collides with the built-in
# ``evox_etl.operators.sampling`` submodule -- it must be MERGED, not clobbered.
COLLIDE_SOURCE = textwrap.dedent(
    '''
    """Synthetic extension colliding with ``evox_etl.operators.sampling``."""


    def probe_helper(value):
        """Marker function lifted onto the pre-existing sampling module."""
        return value
    '''
)


def _target_modules() -> dict[str, types.ModuleType]:
    """The five ``evox_etl`` domain modules plus their immediate submodules."""
    found: dict[str, types.ModuleType] = {}
    for domain in DOMAINS:
        module = importlib.import_module(f"evox_etl.{domain}")
        found.setdefault(module.__name__, module)
        for value in module.__dict__.values():
            if isinstance(value, types.ModuleType) and value.__name__.startswith("evox_etl."):
                found.setdefault(value.__name__, value)
    return found


def _snapshot(module: types.ModuleType):
    """Snapshot a module's attribute bindings and its (copied) ``__all__``."""
    all_before = None if "__all__" not in module.__dict__ else list(module.__dict__["__all__"])
    return dict(module.__dict__), all_before


def _restore(module: types.ModuleType, snapshot) -> None:
    """Undo any attribute/``__all__`` change made since ``_snapshot``."""
    values, all_before = snapshot
    for key in list(module.__dict__):
        if key not in values:
            del module.__dict__[key]
    for key, value in values.items():
        if module.__dict__.get(key) is not value:
            module.__dict__[key] = value
    if all_before is None:
        module.__dict__.pop("__all__", None)
    else:
        module.__dict__["__all__"] = all_before


def _purge_ext_modules() -> None:
    """Drop every ``evox_etl_ext`` entry from ``sys.modules``."""
    for name in [n for n in list(sys.modules) if n == "evox_etl_ext" or n.startswith("evox_etl_ext.")]:
        del sys.modules[name]


@pytest.fixture
def synthetic_ext(tmp_path) -> Callable:
    """Yield a writer for synthetic ``evox_etl_ext/<domain>/<name>.py`` modules.

    ``tmp_path`` becomes another path entry of the ``evox_etl_ext`` namespace
    package, so a module written through the returned writer is discovered
    exactly like an installed extension. The ``evox_etl`` target modules are
    snapshotted and restored, and ``sys.path``/``sys.modules`` are reverted, on
    teardown.
    """
    modules = _target_modules()
    snapshots = {name: _snapshot(module) for name, module in modules.items()}

    sys.path.insert(0, str(tmp_path))
    _purge_ext_modules()
    importlib.invalidate_caches()

    def write(name: str, source: str, domain: str = DOMAIN) -> None:
        """Write ``source`` as ``evox_etl_ext/<domain>/<name>.py``."""
        root = tmp_path / "evox_etl_ext" / domain
        root.mkdir(parents=True, exist_ok=True)
        (root / f"{name}.py").write_text(source, encoding="utf-8")
        importlib.invalidate_caches()

    try:
        yield write
    finally:
        for name, module in modules.items():
            _restore(module, snapshots[name])
        _purge_ext_modules()
        sys.path.remove(str(tmp_path))
        importlib.invalidate_caches()


def test_no_extensions_is_a_clean_noop():
    """With nothing installed the loader does not raise and changes nothing."""
    before = {
        domain: list(importlib.import_module(f"evox_etl.{domain}").__all__)
        for domain in DOMAINS
    }

    auto_load_extensions()
    auto_load_extensions()  # a second no-op call must also be harmless

    algorithms = importlib.import_module("evox_etl.algorithms")
    assert hasattr(algorithms, "make_de")
    sampling = importlib.import_module("evox_etl.operators.sampling")
    assert callable(sampling.uniform_sampling)
    for domain in DOMAINS:
        module = importlib.import_module(f"evox_etl.{domain}")
        assert list(module.__all__) == before[domain]


def test_discovers_and_merges_synthetic_extension(synthetic_ext):
    """A synthetic extension module is attached to its domain and exported."""
    synthetic_ext(MOD_NAME, EXT_SOURCE)

    auto_load_extensions()

    target = importlib.import_module(f"evox_etl.{DOMAIN}")
    # the module landed on evox_etl.<domain>
    assert MOD_NAME in vars(target)
    ext_module = getattr(target, MOD_NAME)
    assert ext_module.__name__ == QUALIFIED

    # ... and its name joined the domain's __all__ exactly once
    assert MOD_NAME in target.__all__
    assert target.__all__.count(MOD_NAME) == 1

    # the extension's lifted function + frozen dataclass are reachable / functional
    config = ext_module.make_probe(7)
    assert ext_module.ProbeConfig is type(config)
    assert dataclasses.is_dataclass(config)
    assert type(config).__dataclass_params__.frozen
    assert config.value == 7

    # the built-in submodules were not disturbed by the attach
    assert callable(target.sampling.uniform_sampling)


def test_colliding_submodule_is_merged_not_clobbered(synthetic_ext):
    """A leaf matching a built-in submodule is merged into it, not replaced."""
    synthetic_ext(COLLIDING, COLLIDE_SOURCE)

    auto_load_extensions()

    sampling = importlib.import_module("evox_etl.operators.sampling")
    # the built-in sampling module survives, with its own API intact
    assert callable(sampling.uniform_sampling)
    assert callable(sampling.grid_sampling)
    # the extension's function was lifted into (not over) the existing module
    assert sampling.probe_helper(3) == 3
    assert "probe_helper" in sampling.__all__


def test_auto_load_extensions_is_idempotent(synthetic_ext):
    """Calling the loader twice must not duplicate __all__ or re-attach."""
    synthetic_ext(MOD_NAME, EXT_SOURCE)

    auto_load_extensions()
    auto_load_extensions()

    target = importlib.import_module(f"evox_etl.{DOMAIN}")
    assert target.__all__.count(MOD_NAME) == 1
    assert len(target.__all__) == len(set(target.__all__))
    assert getattr(target, MOD_NAME).__name__ == QUALIFIED

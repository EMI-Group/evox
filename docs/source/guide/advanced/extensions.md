# Extensions

EvoX 2.0 lets a third-party package add **algorithms, problems, operators, metrics or utilities** to `evox` without forking or patching the core. Extensions are discovered at `import evox` time through the PEP 420 namespace package `evox_ext` and merged into the matching `evox.<domain>` module, so afterwards they look exactly like built-ins.

```{warning}
This page describes an **experimental EvoX 2.0 pre-release**. `evox` is the functional rewrite running on the `evoxc` tensor-graph compiler; the 1.x object-oriented API (`ModuleBase`, `Parameter`/`Mutable`, `torch.compile`/`vmap`) was removed in 2.0 and is **not** what extensions target here. The extension namespace is `evox_ext` — the same import name the 1.x plugin namespace used, now serving the functional contract below. Expect the details below to move before 2.0 is final.
```

## How discovery works

Everything lives in `src/evox_ext/autoload_ext.py`. The entry point is called once, at the very end of `src/evox/__init__.py`, inside a guard:

```python
try:
    from evox_ext.autoload_ext import auto_load_extensions

    auto_load_extensions()
except ImportError:
    pass
```

A missing extension namespace is therefore a **silent no-op** — importing `evox` never fails because nothing is installed.

`auto_load_extensions()` walks the five domains in a fixed order:

```python
DOMAINS: tuple[str, ...] = ("utils", "algorithms", "problems", "operators", "metrics")
```

For each domain it (1) imports the target `evox.<domain>` and the extension `evox_ext.<domain>` in one guarded block — an `ImportError` on either side just `continue`s to the next domain — then (2) hands the pair to `load_extension(ext_pkg, target_module)`, which:

1. iterates the extension package's submodules with `pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")` (via the `iter_namespace` helper, so names come back absolute);
2. imports each one with `importlib.import_module`;
3. attaches it on the target module under its **leaf** name (`evox_ext.operators.my_probe` becomes `evox.operators.my_probe`) and appends that leaf name to the target's `__all__`;
4. repeats all of the above recursively when the leaf names another package inside the extension.

Because the loader imports `evox.<domain>` for every domain, the five domain modules are always loaded as attributes of the `evox` package after autoload, even for domains nobody extended.

## Merge semantics

A leaf that **collides** with an existing built-in submodule is merged into it, never substituted. An extension module named `sampling` under `operators`, for example, does not replace `evox.operators.sampling`: its top-level functions and classes are lifted onto the existing module and their names join that submodule's `__all__`, alongside `uniform_sampling` and friends. The extension module object itself stays reachable only as `evox_ext.operators.sampling`.

Two consequences worth knowing:

- Top-level functions and classes defined directly in the extension package (or, in the collision path, in the colliding module) are lifted the same way. The lift walks `dir(...)` and keeps every `inspect.isfunction`/`inspect.isclass` member, so **imported** Python functions get lifted too — import into an extension module only what you mean to export.
- The extension module's own `__all__` is not consulted; `__all__` entries are appended per name actually lifted.

The loader is fully **idempotent**: a second `auto_load_extensions()` re-attaches nothing new and adds no duplicate `__all__` entry (`_extend_all` checks membership first). `DOMAINS` only fixes *which* domain is processed in which order — each domain is merged into its own `evox.<domain>` target module with plain `setattr`, so there is no precedence between domains (a name under `operators` never competes with one under `metrics`). Within a domain, a colliding submodule is merged rather than replaced, and a lifted function or class whose name already exists is simply overwritten by whichever extension is loaded last, since `pkgutil.iter_modules` makes no ordering guarantee.

## The extension contract

Extensions are **functional**, exactly like the core: there are no `ModuleBase`-style classes with behaviour or state in 2.0. A module exposes

- plain module-level functions following the step protocol — `init(config, key) -> state`, `step(config, state, evaluate) -> state`, with optional `init_step`/`final_step` (see `src/evox/core/algorithm.py` for the documentation-only protocol). Functions must be plain functions, not `@evoxc.defn` objects, and only run inside an active `evoxc` trace;
- frozen config dataclasses holding **only plain static leaves** (ints, floats, strings, tuples of floats — never arrays or tensors);
- a `make_*` constructor that validates and normalizes user arguments into that config (see `make_de`, `make_unit_clip`).

For a non-algorithm domain the surface is even smaller: an operator or a metric is just a function (key-first when it draws randomness, e.g. `latin_hypercube_sampling(key, n, d)`), optionally accompanied by a frozen config. See {doc}`operators` for the operator conventions and {doc}`backends` for what those functions are traced onto.

## Package layout and packaging

```
my-evolution-ext/
├── pyproject.toml
└── evox_ext/                # namespace dir — NO __init__.py
    ├── operators/             # namespace dir — NO __init__.py
    │   └── my_probe.py        # plain module: functions + frozen config + make_*
    └── algorithms/
        └── my_ga.py
```

**No `__init__.py` anywhere in the namespace directories** — neither in `evox_ext/` nor in `evox_ext/<domain>/`. That is what makes PEP 420 discovery work: the domain directory becomes one more entry on the namespace package's `__path__`, so `pkgutil.iter_modules` lists your modules next to everyone else's.

For packaging, the only requirement is that the distribution **ships the namespace package**: either a `[tool.setuptools.packages.find]`-style discovery with namespaces enabled (`namespaces = true`), or an explicit `packages` entry that includes `evox_ext.*`. Let `setuptools` discover the tree rather than hand-listing modules, and never add an `__init__.py` to make an installer happy.

## Worked example: a probe operator

`evox_ext/operators/my_probe.py`, using only the functional contract:

```python
"""Probe operator extension: clamp a keyed jitter of ``x`` into a static box."""
import dataclasses

import evoxc
import evoxc.random as random

__all__ = ["UnitClip", "make_unit_clip", "clip_unit"]


@dataclasses.dataclass(frozen=True)
class UnitClip:
    """Frozen config: plain static leaves only."""
    lo: float = -1.0
    hi: float = 1.0
    jitter: float = 0.1


def make_unit_clip(lo=-1.0, hi=1.0, jitter=0.1):
    """Validate + build the frozen config (host-side, ``make_*`` contract)."""
    if not lo < hi:
        raise ValueError("lo must be smaller than hi")
    if jitter < 0.0:
        raise ValueError("jitter must be non-negative")
    return UnitClip(lo=lo, hi=hi, jitter=jitter)


def clip_unit(key, x, config):
    """Operator: jitter ``x`` with a keyed draw, then clamp into the box."""
    _key, subkey = random.split(key)
    noise = random.uniform(subkey, x.shape, low=-config.jitter, high=config.jitter, dtype=x.dtype)
    lo = evoxc.constant(evoxc.core.tensor(config.lo, dtype="float32"))
    hi = evoxc.constant(evoxc.core.tensor(config.hi, dtype="float32"))
    return evoxc.clamp(x + noise, lo, hi)
```

Put the extension tree on `sys.path` **before** importing `evox` (that is the only ordering constraint), and the module is already there:

```python
import sys
sys.path.insert(0, "extroot")            # the directory holding evox_ext/

import evox
import evox.operators as operators

probe = operators.my_probe
assert probe.__name__ == "evox_ext.operators.my_probe"
assert "my_probe" in operators.__all__

config = probe.make_unit_clip(lo=-0.5, hi=0.5)   # host side, no trace needed
```

The operator itself only runs inside a trace — same rule as every built-in operator — so trace it like one:

```python
import numpy as np
import evoxc

exe = evoxc.build(
    lambda key, x: probe.clip_unit(key, x, config),
    evoxc.core.TensorSpec((), np.int64),        # the key
    evoxc.core.TensorSpec((4,), np.float32),    # x
    backend="numpy",
    device=evoxc.core.Device("cpu"),
)
out = evoxc.run(exe, evoxc.random.key(0), np.array([2.0, -2.0, 0.0, 0.1], dtype=np.float32))
print(out.numpy())   # every entry clamped into [-0.5, 0.5]
```

Calling `auto_load_extensions()` again afterwards leaves `evox.operators.__all__` unchanged (no duplicates), which is what makes repeated imports and re-testing harmless.

```{note}
An extension **algorithm** must define `init`/`step` in the *same module* as its config dataclass: `StdWorkflow` resolves the implementation through `importlib.import_module(type(config).__module__)`. A config imported from a `config.py` next to the algorithm will not find its `step`.
```

## How autoloading is tested

`unit_test/evox/ext/test_autoload.py` proves the guard is a genuine no-op in a *fresh interpreter*: it spawns a subprocess that poisons `sys.modules["evox_ext"]` so any extension import raises `ImportError`, then imports `evox` and checks it still succeeds.

`unit_test/evox/ext/test_autoload_ext.py` builds a synthetic `evox_ext/<domain>/` namespace under `tmp_path`, appends it to `sys.path`, and asserts the three behaviours: discovery (module attached under its leaf name, name in `__all__` exactly once, frozen dataclass usable), merge-into-existing-module (a colliding `sampling` extension leaves `uniform_sampling`/`grid_sampling` intact while adding `probe_helper`), and idempotency. Run just this suite with:

```bash
python -m pytest unit_test/evox/ext -q
```

```{tip}
Extensions inherit the `evoxc` trace constraints wholesale: functions are stateless, shapes are static, and randomness is keyed (`random.split` from a key carried in the state). An extension can never fall back to eager tensor ops — there is no eager mode — so prototype inside `evoxc.build`/`evoxc.run` from the start. See {doc}`../developer/architecture` and {doc}`../developer/custom-alg-pro`.
```

# evox_etl_ext — Extension Autoloading for the ETL Rewrite

## Intent
Plugin/extension system for the functional ETL port (`evox_etl`), mirroring
`src/evox_ext/` (the torch variant). Third-party packages register new ETL
algorithms, problems, operators, metrics or utilities without touching the core
`evox_etl` package. Extensions are discovered at load time through Python's
**PEP 420 namespace packages**.

The ETL model is purely functional (plain functions + frozen config dataclasses
+ `make_*` constructors — no `ModuleBase` classes; see
`../evox_etl/DESIGN.md`), so an ETL extension module exposes exactly the same
surface as a built-in `evox_etl` module.

## API Surface
- **`autoload_ext.auto_load_extensions() -> None`** — Sole public entry point.
  Called once from `evox_etl/__init__.py`. Iterates the 5 domains and merges
  every installed extension into the matching `evox_etl.<domain>` module.
- **`autoload_ext.load_extension(ext_pkg, target_module) -> None`** — Recursive
  internal loader. For one `evox_etl_ext.<domain>` namespace it:
  1. iterates submodules via `pkgutil.iter_modules` and imports each;
  2. attaches each under its leaf name on `target_module`;
  3. if a same-named module already exists on the target, merges into it
     recursively (never clobbers);
  4. lifts the extension package's own top-level functions/classes onto the
     target.
- **`autoload_ext.iter_namespace(ns_pkg)`** — `pkgutil.iter_modules` wrapper
  yielding absolute submodule names.
- **`autoload_ext.DOMAINS`** — the 5 extension domains, in merge-priority order:
  `("utils", "algorithms", "problems", "operators", "metrics")`.

## Extension Contract (author's view)
An extension module lives at `evox_etl_ext/<domain>/<name>.py` (a `<domain>` dir
with NO `__init__.py`, exactly like the top-level `evox_etl_ext/` namespace dir)
and exposes:

```
evox_etl_ext/
  algorithms/
    my_ga.py        # plain module-level functions + frozen config dataclass + make_*
  operators/
    my_op.py
```

- **NO classes with behaviour / state**: the ETL model has no `ModuleBase`. Provide
  plain functions (`init`, `step`, ... following the step protocol in
  `../evox_etl/core/algorithm.py`) plus frozen config dataclasses and `make_*`
  constructors.
- The module is merged onto the matching `evox_etl.<domain>` package, so
  `from evox_etl.algorithms import *` picks it up (its name is appended to
  `__all__`) and `evox_etl.algorithms.my_ga` resolves after autoload.
- If the leaf name matches an existing built-in submodule (e.g. a module named
  `sampling` under `operators`), the extension is merged INTO it: its top-level
  functions/classes are lifted onto the existing submodule and its names join
  that submodule's `__all__`. It never replaces the built-in module.

## Constraints
- **No `__init__.py` anywhere under this tree** — `evox_etl_ext`, and its
  extension-domain subdirs, are PEP 420 namespace packages. This directory
  holds only the loader (`autoload_ext.py`); it is machinery, not a home for
  real extensions.
- **Do NOT call `auto_load_extensions()` here** — the call belongs in
  `evox_etl/__init__.py` (owned elsewhere).
- Every domain is wrapped in `try/except ImportError`: a domain with no
  extension (or a missing target module) is skipped silently, so the loader is a
  clean no-op when nothing is installed.
- **Fully idempotent**: a second `auto_load_extensions()` re-attaches nothing and
  appends no duplicate `__all__` entries.
- Merge priority follows `DOMAINS` order: a later domain never overrides an
  earlier one at the same level — only genuinely new names are added; a collision
  is merged, never clobbered.

## Routing Table
| Path | Description |
|---|---|
| `autoload_ext.py` | The sole loader file: discovery, recursive merge, `auto_load_extensions()` |
| `../evox_etl/` | Functional package the extensions merge INTO |
| `../evox_ext/` | Torch-variant reference implementation of this mechanism (READ-ONLY) |
| `../../unit_test/etl/ext/` | Tests for this loader (no-op, discovery/merge, idempotency) |

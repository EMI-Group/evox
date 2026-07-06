# evox_ext — Extension Autoloading Mechanism

## Intent
Provides the plugin/extension autoloading system for EvoX. External packages can register new algorithms, problems, operators, metrics, or utilities without modifying the core EvoX codebase. Extensions are discovered at import time via Python's **PEP 420 namespace packages**.

## API Surface
- **`autoload_ext.auto_load_extensions()`** — The sole public entry point. Called once during `import evox` (from `evox/__init__.py`). Discovers and merges all extensions from the `evox_ext` namespace into the corresponding `evox.*` modules.
- **`autoload_ext.load_extension(package, exposed_module)`** — Recursive internal loader. For a given `evox_ext.<domain>` namespace, it:
  1. Iterates all installed sub-packages via `pkgutil.iter_modules`.
  2. Recursively merges matching module trees.
  3. Lifts top-level classes and functions directly onto `exposed_module`.

## How Extensions Work

### Registration (external package author)
A third-party package installs modules into the `evox_ext` namespace — for example:

```
evox_ext/
  algorithms/
    my_custom_ga.py      # defines a class inheriting from evox.core.ModuleBase
  problems/
    my_problem.py         # defines a problem class
```

No `__init__.py` files are needed; the `evox_ext` namespace is shared across all installed packages that contribute to it.

### Discovery & Merging (at import time)
When `import evox` runs, `auto_load_extensions()` attempts to import each of the 5 extensible domains:

| Namespace Package       | Target Module        |
|------------------------|---------------------|
| `evox_ext.utils`       | `evox.utils`        |
| `evox_ext.algorithms`  | `evox.algorithms`   |
| `evox_ext.problems`    | `evox.problems`     |
| `evox_ext.operators`   | `evox.operators`    |
| `evox_ext.metrics`     | `evox.metrics`      |

Each domain is wrapped in a `try/except ImportError` — if no extension exists for a domain, it silently passes.

The recursive merge algorithm:
- **Module-level match**: If `evox_ext.algorithms.foo` corresponds to an existing `evox.algorithms.foo`, recurse into it.
- **No match**: The entire external module is attached as a new attribute on the target.
- **Class/function lift**: Classes and functions defined directly in the namespace package are promoted onto the target module.

After loading, `exposed_module.__all__` is updated so that `from evox.algorithms import *` includes extension classes.

## Constraints
- This directory should remain minimal — it is purely the loading machinery, not a place to put actual extensions.
- External packages must use the PEP 420 namespace layout (no `__init__.py` in `evox_ext/`).
- Extensions are loaded at import time, not lazily. Heavy initialization should be deferred inside the extension module.
- The import order in `auto_load_extensions()` determines merge priority: later domains don't override earlier ones at the same level (only new modules are added).

## Routing Table
- `autoload_ext.py` → The sole file; contains all logic for discovery, recursive merging, and the public `auto_load_extensions()` entry point.

# docs/source/guide/developer/ — EvoX Developer Guide

## Intent
This directory contains the **developer-oriented guide pages** for contributing to the EvoX framework. The content targets developers who are setting up their environment, building custom algorithms/problems, working with the ModuleBase API, deploying hyperparameter optimization (HPO), or writing documentation. All pages are in English.

## API Surface

The developer guide is integrated into the documentation via `index.md`, which serves as a single toctree landing page pointing to five sub-pages:

```{toctree}
:maxdepth: 1
environment
modulebase
custom-alg-pro
custom_hpo_prob
document
```

### Page-by-page summary

| File | Format | Purpose |
|---|---|---|
| `index.md` | MyST Markdown | TOC landing page; `{toctree}` directive linking all five sub-pages |
| `environment.md` | MyST Markdown | Dev environment: cloning the repo, editable install `pip install -e ".[test]"`, Nix shell (`nix develop .`), ruff linting, pre-commit hooks, running unit tests with `python -m unittest` |
| `modulebase.md` | MyST Markdown | Deep-dive into `ModuleBase` (`evox.core.module.ModuleBase`, inherits `torch.nn.Module`): key methods (`load_state_dict`, `add_mutable`), role (mutable values, functional programming via `state_dict`/`load_state_dict`, standardized init via `setup`), static (`@jit`) vs. non-static method rules, and using `torch.cond` for vmap-compatible control flow |
| `custom-alg-pro.md` | MyST Markdown | Step-by-step guide to implementing custom `Algorithm` and `Problem` classes. Covers: flat layout (`Algorithm.step` ↔ `Problem.evaluate`), required/optional method tables, `setup` vs `__init__` initialization rules, and a full end-to-end PSO + Sphere example with complete code |
| `custom_hpo_prob.ipynb` | Jupyter Notebook | Deploying HPO with custom algorithms. Covers: making algorithms parallelizable (no in-place ops, no Python control flow, use `torch.cond`), `HPOMonitor`/`HPOFitnessMonitor` for HPO metric tracking, `HPOProblemWrapper` to convert a `StdWorkflow` into an HPO problem, custom hyperparameter sets, and an outer PSO workflow that optimizes inner algorithm hyperparameters |
| `document.md` | MyST Markdown | Documentation writing conventions: Sphinx-style docstrings without types in docstrings (`:param`, `:return`, `:raises`), external docs formats (`.md` preferred for narrative, `.ipynb` for interactive content), notebook execution rules (pre-execute before commit; no GPU in CI), cross-reference syntax (`[label](#ref)`), image embedding (`![Alt](path)`), and translation workflow commands (`make gettext`, `sphinx-intl update`, `fix_output.py`) |

## Constraints

- All pages are **hand-written** — no auto-generated content here (autodoc-generated API reference lives under `source/apidocs/`)
- Notebook (`custom_hpo_prob.ipynb`) is **pre-executed with output cells saved**; CI has no GPU and cannot execute it
- Internal cross-references use Sphinx/MyST `[label](#target)` syntax, pointing into the codebase or other doc pages
- Images use paths relative to `source/`, e.g., `/_static/modulebase.png`
- Docstrings in source code are covered by `document.md`'s conventions; the actual code lives in `src/evox/`
- No child directories — all pages are flat files, so there is no further routing to delegate

## Routing Table

This is a leaf node with no child directories. All guide content is contained in the six files listed above.

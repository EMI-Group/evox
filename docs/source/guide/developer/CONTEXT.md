# developer/ — Developer Guide

## Intent
This directory holds the **Developer Guide** section of the EvoX documentation. It covers everything contributors need: setting up a dev environment, understanding `ModuleBase` (the foundational class), implementing custom algorithms and problems, deploying HPO with custom algorithms, and writing documentation. The section is linked from the root landing page (`source/index.md`) under the "Getting Started" toctree and as a grid-item-card labeled "Developer Guide."

## API Surface

| File | Page Title | Description |
|---|---|---|
| `index.md` | Developer Guide | Section landing page; toctree at `:maxdepth: 1` linking all five sub-pages |
| `environment.md` | Develop environment | Clone repo, pip editable install with test deps, Nix shell, ruff linting, pre-commit setup, and `python -m unittest` usage |
| `modulebase.md` | Working with Module in EvoX | Introduces `ModuleBase` (extends `torch.nn.Module`): key methods (`__init__`, `load_state_dict`, `add_mutable`), its role (mutable values, functional programming via state dicts, standardized init via `setup`), and usage rules (`@jit` for static methods, `torch.cond` for vmap-compatible control flow) |
| `custom-alg-pro.md` | Custom algorithms and problems in EvoX | Explains the flat Algorithm–Problem layout; documents the `Algorithm` interface (`__init__`, `step`, optional `init_step`) and `Problem` interface (`__init__`, `evaluate`); provides complete code examples: PSO algorithm + Sphere problem |
| `custom_hpo_prob.ipynb` | Deploy HPO with Custom Algorithms | Jupyter notebook; covers making algorithms parallelizable for HPO (no in-place operations, no Python control flow), and wrapping custom algorithms as HPO problems |
| `document.md` | Document Writing Guide | Sphinx-style docstring conventions (`:param`, `:return`, `:raises`), Markdown/IPYNB preferences, pre-execution requirement for notebooks, cross-reference syntax, and translation update workflow (`make gettext` → `sphinx-intl update` → `fix_output.py`) |

## Constraints

- The `.ipynb` file (`custom_hpo_prob.ipynb`) must be **pre-executed and saved with output** before committing, since the build system has `nb_execution_mode = "off"` — CI/CD cannot run GPU-dependent cells
- All pages use **MyST Markdown** (`.md`) rendered by Sphinx with the `shibuya` theme, except the notebook which is processed by the `myst_nb` extension
- Cross-references use the `[label](#ref)` syntax (e.g., `[ModuleBase](#evox.core.module.ModuleBase)`)
- Docstrings in the EvoX source code follow **Sphinx-style** (no types in docstrings — types are in function signatures)

## Routing Table

This directory is a leaf node — there are no child subdirectories. All six files are at the top level and listed above.

# guide/ — Getting Started Guides

## Intent
This directory groups the three **Getting Started** guide sections for EvoX, linked from the root documentation landing page (`source/index.md`) under the "Getting Started" toctree and as grid-item-cards. There is **no top-level `guide/index.md`** — each child subdirectory has its own `index.md` that serves as the landing page for that section.

## API Surface

| File / Directory | Purpose |
|---|---|
| `install/index.md` | Section landing: "Installation Guide" — toctree links to `python` and `torch_and_evox` |
| `developer/index.md` | Section landing: "Developer Guide" — toctree links to `environment`, `modulebase`, `custom-alg-pro`, `custom_hpo_prob`, `document` |
| `experimental/index.md` | Section landing: "Experimental Features" — toctree links to `distributed_workflow` |

## Constraints
- There is no `guide/index.md`; the root `source/index.md` links directly to `guide/install/index`, `guide/developer/index`, and `guide/experimental/index` via the toctree and grid-item-cards
- All three sections share the `:maxdepth: 1` toctree setting
- The `.ipynb` file (`developer/custom_hpo_prob.ipynb`) is rendered as a MyST notebook (via `myst_nb` extension); it must be pre-executed before committing since notebook execution is disabled at build time

## Routing Table

| Area | Child Directory | Description |
|---|---|---|
| Installation Guide | `install/` | Setting up Python, PyTorch, and EvoX |
| Developer Guide | `developer/` | Contributing: dev environment, ModuleBase API, custom algorithms/problems/HPO, documentation |
| Experimental Features | `experimental/` | Multi-GPU and distributed workflow documentation |

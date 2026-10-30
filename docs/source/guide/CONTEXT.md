# guide/ — Getting Started Guides
## Intent
This directory groups the three **Getting Started** guide sections for EvoX 2.0, linked from the root landing page (`source/index.md`) under the "Getting Started" toctree and as grid-item-cards.
Every page here documents the experimental EvoX 2.0 pre-release (import name `evox_etl`, compiled by `evoxc`); the EvoX 1.x torch API is deliberately absent.
There is **no top-level `guide/index.md`** — each child subdirectory has its own `index.md` that serves as the landing page for that section.

## API Surface
| File / Directory | Purpose |
|---|---|
| `install/index.md` | Section landing: "Installation Guide" — toctree links to `python` and `evox` |
| `install/python.md` | Installing and isolating a Python ≥3.11 environment |
| `install/evox.md` | Installing `evox` (import `evox_etl`) as the 2.0.0a1 pre-release or from a checkout, the backend extras, the XLA PJRT plugin, smoke test |
| `developer/index.md` | Section landing: "Developer Guide" — toctree links to `architecture`, `custom-alg-pro`, `custom-monitor`, `environment`, `document` |
| `developer/architecture.md` | Functional architecture: config/state split, step protocol, compile-once workflow, `make_*` convention |
| `developer/custom-alg-pro.md` | Authoring a custom algorithm and a custom problem (frozen config + `init`/`step`/`evaluate`) |
| `developer/custom-monitor.md` | Authoring a custom monitor, the host-side aux-history channel |
| `developer/environment.md` | Contributor setup: editable install, `evoxc`, Nix/pre-commit, canonical `unit_test/etl` suite, docs build |
| `developer/document.md` | Documentation contribution: MyST/Sphinx layout, autodoc2, bilingual gettext policy |
| `advanced/index.md` | Section landing: "Advanced Guide" — toctree links to `backends`, `operators`, `metrics`, `viz`, `hpo`, `extensions` |
| `advanced/backends.md` | Backend selection (`numpy`/`iree`/`tvm`/`xla`), extras, PJRT plugin, performance and limitation notes |
| `advanced/operators.md` | Operator catalog, pure-function contract, `jit_fix_operator` helpers, MO operator injection |
| `advanced/metrics.md` | GD/IGD/HV and the Monte-Carlo HV helpers |
| `advanced/viz.md` | `vis_tools` Plotly builders, `EvalMonitor.plot`, `.exv` serialization |
| `advanced/hpo.md` | Host-side HPO wrapper (`HPOProblemWrapper`, `HPSlot`, `random_search`) |
| `advanced/extensions.md` | `evox_etl_ext` PEP 420 extension autoloading |

## Constraints
- There is no `guide/index.md`; the root `source/index.md` links directly to `guide/install/index`, `guide/developer/index`, and `guide/advanced/index`.
- All three sections share the `:maxdepth: 1` toctree setting.
- No EvoX 1.x API may appear except as an explicit "not covered" statement; snippets must be valid 2.0 `evox_etl` code verified against `src/evox_etl`.
- Distributed-workflow documentation has no 2.0 counterpart (the workflow is single-device) and must not be reintroduced.

## Routing Table
| Area | Child Directory | Description |
|---|---|---|
| Installation Guide | `install/` | Setting up Python, installing `evox`, backend extras, XLA PJRT plugin |
| Developer Guide | `developer/` | Functional architecture, custom algorithm/problem/monitor authoring, contributor environment, documentation |
| Advanced Guide | `advanced/` | Backends, operators, metrics, visualization, host-side HPO, extension system |

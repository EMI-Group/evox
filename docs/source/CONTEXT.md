# docs/source/ — Sphinx Documentation Source Root
## Intent
This directory is the **Sphinx documentation source root** for EvoX and contains everything `sphinx-build` compiles into the published HTML on Read the Docs: configuration, content pages, static assets, templates and i18n resources.
It documents the **experimental EvoX 2.0 pre-release** only: the functional package `src/evox_etl/` (import name `evox_etl`) running on the EvoX Compiler (`evoxc`).
The EvoX 1.x torch package (`src/evox/`, deleted in 2.0) is not documented anywhere; the only place 1.x appears is the migration page `tutorial/tutorial_part7.md`, which maps removed APIs to their 2.0 equivalents.
The documentation is bilingual: English sources here, Simplified Chinese under `locale/`.

## API Surface
| File / Directory | Purpose |
|---|---|
| `conf.py` | Sphinx configuration: extensions, shibuya theme, autodoc2 (scanning `../../src/evox_etl` only), i18n, nav links, `version = "2.0.0-pre"` |
| `index.md` | Root landing page: pre-release warning, functional-model summary, runnable quickstart, feature list, toctrees + grid-card navigation |
| `tutorial/` | 7-part getting-started series (2.0 only) |
| `guide/install/` | Python and `evox` installation, backend extras, XLA PJRT plugin |
| `guide/developer/` | Functional architecture, custom algorithm/problem/monitor authoring, contributor environment, documentation contribution |
| `guide/advanced/` | Backends, operators, metrics, visualization/`.exv`, host-side HPO, `evox_etl_ext` extensions |
| `examples/` | Markdown example pages (`so`, `mo`, `neuroevolution`) with executed outputs |
| `miscellaneous/` | Device selection, non-NVIDIA hardware, Linux distribution notes, MATLAB migration |
| `_static/`, `_templates/` | Assets (logos, CSS, favicons) and Jinja2 templates |
| `locale/` | `zh_CN` gettext catalog + full-page tutorial translations |

## Constraints
- **Only the 2.0 API**: no page may teach `ModuleBase`, `Parameter`/`Mutable`, `compile`/`vmap`/`use_state`, class-based components, `enable_distributed`, the Triton kernel package, Brax/MuJoCo/supervised-learning problems, or `evox.__version__`; such mentions are limited to short "removed in 2.0" statements and the migration page.
- **Every snippet must be true 2.0 code** verified against `src/evox_etl` (imports `evox_etl.*`, config dataclasses + `make_*`, `StdWorkflow`, stateless `key(seed)`); do not invent kwargs.
- **apidocs/ is generated**: `autodoc2` writes it at build time from `src/evox_etl`; it is git-ignored and must never be edited or committed. Delete `source/apidocs/` when the scanned package path changes, otherwise stale pages produce `autodoc2.missing` warnings.
- `autodoc2_module_all_regexes` only lists packages whose every module defines `__all__` (`core`, `problems`, `utils`, `workflows`) — matching a module without `__all__` aborts the build.
- `exclude_patterns` keeps `locale/**`, every `CONTEXT.md`, and one orphan autodoc2 page (`evox_etl.problems.numerical.state`) out of the source scan.
- **Install/backend facts**: 2.0 requires Python 3.11+ and the `evoxc` compiler alongside `evox`; the compiled backends are the extras `evox[iree]` / `evox[tvm]` / `evox[compilers]` (equivalently `evoxc[...]`, which is the spelling that installs under the current 1.x manifest), `evox[vis]` adds Plotly, and XLA needs a user-supplied PJRT plugin via `EVOXC_PJRT_PLUGIN` exported before the process starts; `guide/install/evox.md` owns the install story and `guide/advanced/backends.md` owns backend detail.
- Content pages are MyST Markdown (`{toctree}`, `{note}`/`{warning}`/`{tip}`, `:::` colon fences, `{eval-rst}` grids, `{doc}` roles); notebook execution is disabled (`nb_execution_mode = "off"`), so no page may rely on build-time execution.
- Build check: `PYTHONPATH=../src python -m sphinx -b html source <out>` from `docs/`; a clean tree builds warning-free.
- Count claims stay identical across EN/ZH and match the repository: EvoX core "30+ algorithms" / "30+ benchmark problems"; ecosystem "140+ algorithms" / "240+ benchmark problems/environments".

## Routing Table
| Area | Child | Description |
|---|---|---|
| Sphinx configuration | `conf.py` | Extension loading, shibuya theme, autodoc2 package targets, i18n, nav links |
| Root landing page | `index.md` | Toctrees, quickstart, grid-card navigation |
| Tutorial series | `tutorial/` | 7 parts: install, quickstart, workflow+monitor, algorithms, problems, multi-objective, migration |
| Installation guide | `guide/install/` | Python setup, `evox` install, backend extras, PJRT plugin |
| Developer guide | `guide/developer/` | Functional architecture, component authoring, environment, documentation |
| Advanced guide | `guide/advanced/` | Backends/performance, operators, metrics, viz/.exv, host-side HPO, extensions |
| Examples | `examples/` | Single-objective, multi-objective and neuroevolution walkthroughs with real outputs |
| Miscellaneous | `miscellaneous/` | Device/GPU selection, alternative hardware, Linux notes, MATLAB migration |
| Static assets | `_static/` | Logos, CSS (`css/custom.css`), favicons, SVGs, images |
| Templates | `_templates/` | Class/module RST templates and HTML partials |
| i18n | `locale/` | `zh_CN` gettext catalog + partial manual tutorial translations |

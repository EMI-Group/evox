# advanced/ — Advanced Guide (EvoX 2.0 pre-release)

## Intent
The advanced guide for the experimental EvoX 2.0 pre-release: it covers the machinery around the optimizer loop — backends, operators, metrics, visualization, host-side HPO and the extension system.
Every page teaches the functional `evox_etl` API on the `evoxc` compiler and deliberately never teaches the frozen 1.x torch API (`src/evox/`).
The section landing page `index.md` is the single entry point; the root `docs/source/index.md` already lists `guide/advanced/index` in its toctree and card grid.

## Page map
| Page | Content |
|---|---|
| `index.md` | Landing page, `{toctree}` (`:maxdepth: 1`) over the six content pages, experimental-pre-release warning, per-page summary table, shared conventions |
| `backends.md` | NumPy interpreter vs compiled IREE/TVM/XLA, pip extras, XLA PJRT plugin via `EVOXC_PJRT_PLUGIN`/`plugin_path`, `backend=`/`device=`/`compile_options=` reaching `evoxc.build`, workload-to-backend table, REFACTOR_REPORT §6 limitations, bit-identical results across compiled backends |
| `operators.md` | `sampling`/`selection`/`crossover`/`mutation` catalog with source signatures, pure-function contract, `jit_fix_operator` helpers, traced usage example, MO `selection_op`/`crossover_op`/`mutation_op` injection |
| `metrics.md` | `gd`, `gd_plus`, `igd`, `igd_plus`, `hv` (leading RNG key) and the two Monte-Carlo HV estimators, traced-call idiom, worked IGD/HV example on NSGA2+DTLZ2, reference-point guidance |
| `viz.md` | `evox_etl.vis_tools` Plotly builders, `EvalMonitor.plot(problem_pf=, source=)`, EvoXVision `.exv` byte layout, writer-only `EvoXVisionAdapter` plus a hand-parsed read-back round-trip |
| `hpo.md` | Host-side HPO: `HPOProblemWrapper`, `HPSlot`, `HPOProblemConfig`, `HPOFitnessMonitor(Config)`, `random_search`; why nesting is impossible; limitations |
| `extensions.md` | `evox_etl_ext` PEP 420 namespace, import-time `auto_load_extensions()`, merge semantics, functional extension contract, minimal package layout, autoload tests |

## Constraints
- Every runnable Python block on every page was executed against this repository on the NumPy backend (`PYTHONPATH=src /tmp/venv/bin/python`); keep it that way when editing.
- Three `backends.md` blocks are illustrative fragments or compiled-backend-only and cannot be executed as written (undefined `algorithm`/`problem`, an abridged `_build_step_exe` excerpt, an `iree`/`xla` re-run).
- `evoxc` has no eager mode: examples must trace through `evoxc.build` + `evoxc.run` (or a `StdWorkflow` step) and use `.numpy()` on results — never `np.asarray` on a `Tensor`.
- Compiled backends are opt-in: `pyproject.toml` defines only the `vis`, `neuroevolution`, `test`, `docs`, `default` and `triton` extras — there is no `iree`, `tvm`, `compilers` or `cuda` extra; XLA has no extra and needs a user-supplied PJRT plugin, so no page here claims to have run a compiled backend.
- Performance statements are quoted only from `REFACTOR_REPORT.md` §3/§6 and `benchmarks/etl_vs_torch/`, never invented.
- One H1 per page, MyST Markdown, and the shared `{warning}` block marking the experimental 2.0 pre-release on every content page.
- Cross-references use agreed paths (`../install/evox`, `../install/python`, `../developer/architecture`, `../developer/custom-alg-pro`, `../../tutorial/tutorial_part2`…`part7`, sibling `{doc}` links); pages owned by other agents may not exist yet.
- Optional extras stay opt-in: examples run with no toolchain, GPU driver or PyTorch installed.
- Never edit `docs/source/apidocs/` (generated, git-ignored) and never touch `conf.py`, `docs/source/index.md`, `locale/` or `src/` from here.

# advanced/ — Advanced Guide (EvoX 2.0 pre-release)

## Intent
The advanced guide for the experimental EvoX 2.0 pre-release: it covers the machinery around the optimizer loop — backends, operators, metrics, visualization, host-side HPO and the extension system.
Every page teaches the functional `evox` API on the `evoxc` compiler and deliberately never teaches the 1.x torch API (removed in 2.0).
The section landing page `index.md` is the single entry point; the root `docs/source/index.md` already lists `guide/advanced/index` in its toctree and card grid.

## Page map
| Page | Content |
|---|---|
| `index.md` | Landing page, `{toctree}` (`:maxdepth: 1`) over the six content pages, experimental-pre-release warning, per-page summary table, shared conventions |
| `backends.md` | NumPy interpreter vs compiled IREE/TVM/XLA, pip extras (`evox[...]` forwards to `evoxc[...]`, deferring to `../install/evox`), XLA PJRT plugin via `EVOXC_PJRT_PLUGIN`/`plugin_path`, `backend=`/`device=`/`compile_options=` reaching `evoxc.build`, workload-to-backend table, REFACTOR_REPORT §6 limitations, scoped result agreement (IREE bit-identical to numpy; XLA fp32 `dot` may use TF32; TVM CPU-only) |
| `operators.md` | `sampling`/`selection`/`crossover`/`mutation` catalog with source signatures, pure-function contract, `jit_fix_operator` helpers, traced usage example, MO `selection_op`/`crossover_op`/`mutation_op` injection |
| `metrics.md` | `gd`, `gd_plus` (the `max(pf - obj, 0)` direction, documented truthfully), `igd`, `igd_plus`, `hv` (leading RNG key) and the two Monte-Carlo HV estimators, traced-call idiom, worked IGD/HV example on NSGA2+DTLZ2, reference-point guidance |
| `viz.md` | `evox.vis_tools` Plotly builders, `EvalMonitor.plot(problem_pf=, source=)`, EvoXVision `.exv` byte layout, writer-only `EvoXVisionAdapter` plus a hand-parsed read-back round-trip |
| `hpo.md` | Host-side HPO: `HPOProblemWrapper`, `HPSlot`, `HPOProblemConfig`, `HPOFitnessMonitor(Config)`, `random_search`; why nesting is impossible; limitations |
| `extensions.md` | `evox_ext` PEP 420 namespace, import-time `auto_load_extensions()`, per-domain `setattr` merge semantics, functional extension contract, minimal package layout, autoload tests |

## Known Issues
- The `evox[iree]`/`evox[tvm]`/`evox[compilers]` extras forward to the compiler's `evoxc[iree]`/`evoxc[tvm]`/`evoxc[compilers]`, so both spellings install the same toolchain; `backends.md` says so once and defers to `../install/evox`.
- Three `backends.md` blocks are labelled excerpts and are not executed: the `StdWorkflow` constructor sketch and the IREE/XLA alternatives reuse `algorithm`/`problem`, and the `_build_step_exe` fragment is abridged source with a bare `return`.
- `metrics.md` `gd_plus` uses `max(pf - obj, 0)`: a front point *better* than the solution contributes nothing, so the score is `0` when every solution is weakly dominated by a front point — the opposite direction from textbook GD+, and the page says so.
- The `metrics.md` worked NSGA2+DTLZ2 block emits a benign `RuntimeWarning: divide by zero` from the crowding-distance normalisation; the page notes it, matching `hpo.md`.

## Constraints
- Every standalone Python block on every page was executed against this repository on the NumPy backend (`PYTHONPATH=src /tmp/venv/bin/python`); keep it that way when editing.
- Exactly the three `backends.md` blocks listed above are excerpts (or need a compiled backend) and must stay marked as such, with no quoted output.
- `evoxc` has no eager mode: examples must trace through `evoxc.build` + `evoxc.run` (or a `StdWorkflow` step) and use `.numpy()` on results — never `np.asarray` on a `Tensor`.
- Compiled backends are opt-in: `pyproject.toml` defines the `iree`, `tvm`, `compilers` (each forwarding to the matching `evoxc[...]`), `all`, `vis`, `default`, `test` and `docs` extras; XLA has no extra and needs a user-supplied PJRT plugin.
- No page here claims to have run a compiled backend.
- Performance statements are quoted only from `REFACTOR_REPORT.md` §3/§6 and `benchmarks/evox_vs_torch/`, never invented.
- One H1 per page, MyST Markdown, and the shared `{warning}` block marking the experimental 2.0 pre-release on every content page.
- Cross-references use agreed paths (`../install/evox`, `../install/python`, `../developer/architecture`, `../developer/custom-alg-pro`, `../../tutorial/tutorial_part2`…`part7`, sibling `{doc}` links); pages owned by other agents may not exist yet.
- Optional extras stay opt-in: examples run with no toolchain, GPU driver or PyTorch installed.
- Never edit `docs/source/apidocs/` (generated, git-ignored) and never touch `conf.py`, `docs/source/index.md`, `locale/` or `src/` from here.

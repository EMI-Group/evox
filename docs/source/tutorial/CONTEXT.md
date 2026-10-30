# docs/source/tutorial/ — EvoX 2.0 Getting-Started Tutorial

## Intent
This directory holds the 7-part getting-started tutorial for the **experimental EvoX 2.0 pre-release**, whose import name is `evox_etl` (functional rewrite on the `evoxc` compiler).
The series is the primary onboarding path: install → one complete program → workflow/monitor contract → algorithm catalogue → problem catalogue → multi-objective → migration from 1.x.
Pages are English MyST Markdown; the Chinese translations live in `../locale/zh_CN/tutorial/` and are maintained separately (the machine-translation script skips tutorials).

## API Surface

| File | Title | Content |
|---|---|---|
| `index.md` | EvoX Tutorial | Pre-release `{warning}`, reading-order list-table, `{toctree}` (`:maxdepth: 1`) over the 7 parts, the stack (`evox_etl` / `evoxc` / `evox_etl_ext`), count claims |
| `tutorial_part1.md` | 1. EvoX 2.0 and Installation | One paragraph on what changed vs 1.x, functional model at a glance, install (`pip install evox` = numpy backend, extras `vis`/`iree`/`tvm`/`compilers`, xla needs `EVOXC_PJRT_PLUGIN`), smoke test, next-step links |
| `tutorial_part2.md` | 2. Your First Optimization: PSO on Sphere | Full program adapted from `examples/quickstart.py`, argument table, manual `init`/`init_step`/`step`/`final_step`/`flush` loop, reading results, `monitor.plot()`, one-line algorithm/problem swaps |
| `tutorial_part3.md` | 3. Workflow and Monitor | `StdWorkflow` contract (incl. `solution_transform`/`fitness_transform`/`compile_options`), `run` vs `fit` vs step loop, one step = one generation, `opt_direction`, state threading, stateless RNG, `EvalMonitorConfig` field table, `EvalMonitor` method table, monitors outside the graph |
| `tutorial_part4.md` | 4. Algorithms | Naming rules (bare name = config, ES `*Config` aliases, MO `*Config` only under `.mo`, `make_*` recommended), four executed snippets, one table per family (DE 6 / ES 14 / PSO 7 / MO 6), virtual ES family, MO custom-operator injection, choosing an algorithm |
| `tutorial_part5.md` | 5. Problems | Basic numerical table, DTLZ1–7 (`d`/`m`/`ref_num` + module-level `pf(config)`), CEC2022, shift/affine, virtual problems (`VirtualProblem`/`VirtualLoRAProblem`, `(center_flat, seeds, sigma)` payload), writing your own problem |
| `tutorial_part6.md` | 6. Multi-Objective Optimization | NSGA-II on DTLZ2 with list `opt_direction`, `get_pf()`, the seven metrics (`gd`/`gd_plus`/`igd`/`igd_plus`/`hv` + MC variants), plotting with `problem_pf`, `.exv` export via `EvoXVisionAdapter` |
| `tutorial_part7.md` | 7. Migrating from EvoX 1.x | 10-row mapping table, before/after code pairs, "removed in 2.0" list, migration checklist |

## Constraints
- Every page opens with exactly one H1; the `index.md` H1 is `# EvoX Tutorial` and its toctree must list all seven parts with `:maxdepth: 1`.
- **No 1.x torch API is taught anywhere in this directory except `tutorial_part7.md`** (the migration page); other pages may mention 1.x in at most one "what changed" framing sentence.
- All code blocks are verified against the real `src/evox_etl` package with `PYTHONPATH=src /tmp/venv/bin/python` before being written down; do not publish an unexecuted snippet.
- Guide links use the agreed paths under `../guide/` (`install/index`, `install/evox`, `advanced/backends`, `advanced/operators`, `advanced/metrics`, `advanced/viz`, `advanced/hpo`, `advanced/extensions`, `developer/architecture`, `developer/custom-alg-pro`, `developer/custom-monitor`); some may not exist yet.
- Count claims: EvoX ships **30+ algorithms** and **30+ benchmark problems**; the EMI-Group ecosystem **140+ algorithms** and **240+ benchmark problems/environments** — keep EN/ZH identical.
- Never reference `apidocs/` files as sources (generated at build time) and never edit `locale/**` `.po` files from here.

## Notes for Agents
- `EvalMonitorConfig(pop_size=..., dim=...)` completion: the workflow infers `pop_size`/`dim` only from an algorithm state population (`population`/`pop`); CMA-ES carries no bounds, so its monitor needs an explicit `dim=`.
- Monitors and history are affected by the step-loop style: `run(n)` performs `init_step` + (n-2) plain steps + `final_step` (n total; `n == 1` is only `init_step`), and a manual loop must call `init_step` first; monitor history lists live on the config and keep appending across `run`/`fit` calls on the same workflow (a second `run` doubles the list).
- `wf.flush()` only matters when monitor state lives on a non-CPU device: `_readback` appends CPU leaves immediately, so on the default `backend="numpy"`/`device=None` the history is already complete without flushing.
- `init()` builds only the algorithm/monitor init graphs; the per-variant step graphs are traced lazily on the first `init_step`/`step`/`final_step` call (and re-traced when a state leaf changes shape/dtype).
- Problems and metrics are traced `evoxc` functions: outside a trace use `evoxc.evaluate(fn, *concrete_args)` or `evoxc.build` + `evoxc.run`; `hv` additionally takes a leading RNG key and a static `num_sample`.
- `PSO`/`make_pso` take `lb`/`ub` only (no `dim`; 1-D arrays required), and `DTLZ` exposes its Pareto front as the module-level `pf(config)`, not a method.
- Pre-existing build blocker (outside this node): `autodoc2_module_all_regexes` in `../conf.py` matches `evox_etl.algorithms.*`, which includes `algorithms/so/de_variants/code.py` — a module without `__all__`, so autodoc2 aborts the build at `builder-inited`.

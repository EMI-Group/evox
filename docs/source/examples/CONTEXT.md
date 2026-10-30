# docs/source/examples/ — EvoX 2.0 example pages

## Intent
This directory holds the **example walkthroughs** for the EvoX 2.0 pre-release (import name `evox_etl`, the functional rewrite on the `evoxc` compiler).
Every page is MyST Markdown with runnable, self-contained programs — there are no notebooks here (build-time notebook execution is disabled, so code blocks are never executed by Sphinx).
Each page states what it proves, shows the program, quotes the real printed output, and explains the API surface it touches.

## API Surface
| File | Purpose |
|---|---|
| `index.md` | Landing page: the 2.0 pre-release warning, prerequisites (`PYTHONPATH=src`, optional `plotly`), `{toctree}` (`:maxdepth: 1`) over the three walkthroughs, and the cross-links into `tutorial/` and `guide/`. |
| `so.md` | Single-objective walkthrough: PSO on Sphere mirroring `examples/quickstart.py`, the manual `init`/`step` loop vs `run`, a measured PSO/DE/CMA-ES comparison, `wf.monitor.plot()`, and swapping in `Ackley`/`Rastrigin`. |
| `mo.md` | Multi-objective walkthrough: NSGA-II on DTLZ2 with `opt_direction=["min", "min"]`, `get_pf()` / `get_pf_fitness()`, traced IGD/GD/hypervolume scoring, `plot(problem_pf=...)`, and `.exv` export via `EvoXVisionAdapter`. |
| `neuroevolution.md` | Virtual (training-free) ES: `make_virtual_es` / `make_virtual_lora_es` + `make_virtual_problem`, the network-as-data description (`param_shapes`, `layer_specs`), the `(center_flat, seeds, sigma)` payload protocol, center-network deployment, and why the 1.x Brax/MuJoCo/supervised-learning problems are gone. |

## Constraints
Every code block must be **executed before publication** against the real `src/evox_etl` code (`PYTHONPATH=src python <snippet>.py`), and the quoted numbers must be copied from that run (fixed seed 42).
Document only the 1.x-free surface: `evox_etl` configs built by `make_*`, `StdWorkflow`, `EvalMonitorConfig`, `evox_etl.metrics`, `evox_etl.vis_tools` — never `import evox`, `ModuleBase`, `Parameter`/`Mutable`, `torch.compile`, or the torch neuroevolution problems.
Examples run on the default NumPy backend (`backend="numpy"`); a non-NumPy backend belongs in `tutorial/tutorial_part6`.
`evox_etl.metrics` functions are traced graph functions: examples must build them with `evoxc.build` + `TensorSpec` and pass static scalars to both `build` and `run`.
`EvalMonitorConfig(full_sol_history=True)` is required for `get_pf()`; algorithms with no `population`/`pop` state (CMA-ES) need `EvalMonitorConfig(dim=dim)`.
`hv` takes a leading RNG key (`hv(key, objs, ref, num_sample)`) and a static `num_sample`.
Code blocks use `python` fences only (never executed by the build), with `{note}`/`{tip}`/`{warning}` admonitions and `{doc}` cross-references; one H1 per page.
Count claims stay exactly "30+ algorithms" / "30+ benchmark problems" and, for the ecosystem, "140+ algorithms" / "240+ benchmark problems/environments".
Do not reference the deleted 1.x notebooks, and do not edit `conf.py`, `source/index.md`, `locale/`, `src/`, or other doc subtrees.

## Routing Table
This is a leaf directory: all content is in the four pages listed above.
Related documentation lives elsewhere — `../tutorial/tutorial_part2` (workflow internals), `../tutorial/tutorial_part6` (backends), `../tutorial/tutorial_part7` (1.x migration), `../guide/advanced/metrics`, `../guide/advanced/viz`, `../guide/advanced/hpo`, `../guide/developer/custom-alg-pro`.

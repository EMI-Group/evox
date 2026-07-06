# docs/source/examples/ — Jupyter Notebook Examples

## Intent
This directory contains **Jupyter Notebook examples** showcasing EvoX's capabilities across a spectrum of evolutionary optimization tasks. These notebooks serve as entry-level, hands-on demonstrations for new users exploring the framework. They are rendered as part of the Sphinx documentation via `myst_nb` (execution mode: off).

## API Surface

| File | Purpose |
|---|---|
| `index.md` | MyST toctree that lists all 6 notebooks for Sphinx navigation |
| `so-algorithm.ipynb` | **Numerical Optimization** — Optimizing the Ackley function with PSO using built-in `PSO`, `Ackley`, `StdWorkflow`, and `EvalMonitor` |
| `moalg.ipynb` | **Multi-Objective Optimization** — RVEA on DTLZ2 with IGD metric evaluation, GPU support, and Plotly 3D visualization of Pareto fronts |
| `brax.ipynb` | **Neuroevolution with Brax** — Training a SimpleMLP policy on a Brax "swimmer" environment via PSO, using `ParamsAndVector` to bridge neural network parameters and the evolutionary algorithm |
| `supervised-learning.ipynb` | **Neuroevolution for ML** — MNIST digit classification using PSO + `SupervisedLearningProblem` + `ParamsAndVector`, with optional gradient-based fine-tuning and test evaluation |
| `hpo.ipynb` | **Hyperparameter Optimization** — Using `HPOProblemWrapper` to transform a workflow into a problem, then tuning PSO hyperparameters with another PSO; demonstrates `vmap`-based GPU acceleration |
| `custom_algo_prob.ipynb` | **Custom Algorithm & Problem** — Building a PSO algorithm and a Sphere problem from scratch using `evox.core` primitives (`Algorithm`, `Problem`, `Mutable`, `Parameter`), showing the low-level extension API |

## Integration with Documentation

- **Linked from** `source/index.md` under the "Examples" toctree entry
- **Nav bar entry** configured in `source/conf.py` via `html_theme_options["icon_links"]` pointing to `examples/index`
- All notebooks use **`myst_nb`** for rendering; notebook execution is **disabled at build time** (`nb_execution_mode = "off"`) — notebooks must be pre-executed before committing

## Constraints

- Notebooks must remain **pre-executed** with outputs saved, since Sphinx does not execute them at build time
- Output cells (especially Plotly visualizations and Brax HTML iframes) can be very large; the `.ipynb` files range from ~5 MB to ~11 MB
- Each notebook imports from `evox.*` at the top — these imports must remain valid as the source code evolves
- The toctree in `index.md` uses `:maxdepth: 1` to keep the listing flat; new notebooks should be added to the toctree

## Routing Table

This is a leaf directory with no child subdirectories. All content is in the 7 files listed above.

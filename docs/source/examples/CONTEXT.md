# examples/ — Jupyter Notebook Examples

## Intent
This directory contains **executable Jupyter notebooks** that demonstrate EvoX's capabilities across the major feature domains. The examples serve as the primary hands-on learning resource for users, complementing the tutorial series. Each notebook is a self-contained walkthrough covering a specific use case.

## API Surface

Six notebooks + one index page:

| File | Title | Domain | Focus |
|---|---|---|---|
| `so-algorithm.ipynb` | Numerical Optimization | Single-Objective | PSO optimizing the Ackley function; introduces `EvalMonitor` for tracking fitness |
| `moalg.ipynb` | Multi-Objective Algorithm | Multi-Objective | RVEA on DTLZ2; uses IGD metric + Plotly 3D visualization to show Pareto front convergence |
| `brax.ipynb` | Solving Brax Problems in EvoX | Neuroevolution (RL) | PSO training an MLP policy for the Brax "swimmer" physics environment; demonstrates `ParamsAndVector` adapter and policy visualization |
| `supervised-learning.ipynb` | Neuroevolution for Machine Learning | Neuroevolution (SL) | Neuroevolution for MNIST classification via CNN; uses `SupervisedLearningProblem` and `ParamsAndVector` |
| `hpo.ipynb` | Efficient HPO with EvoX | Hyperparameter Optimization | Meta-optimization: PSO tunes PSO's hyperparameters for a Sphere sub-problem; key components: `HPOProblemWrapper`, `HPOFitnessMonitor`, `Parameter` wrapping |
| `custom_algo_prob.ipynb` | Custom Algorithm and Problem | Extensibility | Building a custom PSO `Algorithm` and Sphere `Problem` from EvoX core primitives (`Algorithm`, `Problem`, `Mutable`, `Parameter`, `jit_class`) |
| `index.md` | — | — | MyST toctree listing all six notebooks |

The notebooks progress naturally from **using built-in components** (so-algorithm, moalg) → **domain-specific applications** (brax, supervised-learning, hpo) → **framework extensibility** (custom_algo_prob).

## Role in Documentation System

- **Build-time behavior**: Notebooks are **NOT executed** at Sphinx build time (`nb_execution_mode = "off"`). They must be pre-executed (outputs committed) before the docs build.
- **Rendering**: Processed by the `myst_nb` extension as MyST Markdown notebooks.
- **Translations**: There is **no Chinese translation** of the examples directory (`locale/zh_CN/examples/` does not exist). The examples are English-only. The machine translation script (`machine_translate.py`) skips tutorials and SVGs, and likely skips these notebooks as well given they have embedded code and output.

## Constraints

- Notebooks **must be pre-executed** before committing. Output cells (including Plotly JSON and Brax HTML viewer) are committed in-repo.
- Large notebooks (~37K lines for `hpo.ipynb`, ~36K for `custom_algo_prob.ipynb`) are dominated by embedded Plotly visualization JSON in output cells.
- All notebooks use the `.venv` kernel; language is Python 3.13+.
- Brax notebook requires the `brax` package (mocked at import level in `conf.py` for autodoc, but needed at execution time).
- The `index.md` toctree controls the ordering and visibility of examples in the documentation sidebar.

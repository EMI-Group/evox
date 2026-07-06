# docs/source/tutorial/ — EvoX Getting-Started Tutorial Series

## Intent
This directory contains the **7-part getting-started tutorial** for EvoX, written in English MyST Markdown. It is the primary onboarding path for new users — starting from "What is EvoX?" through installation, basic usage, advanced features, custom development, troubleshooting, and culminating in practical end-to-end examples.

## API Surface

| File | Purpose |
|---|---|
| `index.md` | Single `{toctree}` listing all 7 parts at `maxdepth: 1`. Serves as the landing page titled "EvoX Tutorial" |
| `tutorial_part1.md` | **1. Introduction** — What is EvoX, key features (modular architecture, distributed execution, functional programming, visualization, extensive algorithm/problem libraries), and use cases (large-scale optimization, multi-objective, HPO, neuroevolution, research) |
| `tutorial_part2.md` | **2. Installation and Environment Setup** — Prerequisites (Python 3.10+, PyTorch), Windows installation (one-click script or manual), Linux installation (pip, Docker/Podman), optional extras (`vis`, `neuroevolution`), verification steps |
| `tutorial_part3.md` | **3. Basic Operations** — First optimization run (PSO + Ackley), the algorithm-problem-workflow-monitor concept, step-by-step project configuration (6-step recipe), basic command reference for `StdWorkflow`, `EvalMonitor`, and device control (`.to()`) |
| `tutorial_part4.md` | **4. Advanced Features** — Custom configuration (algorithm params, operator replacement, multi-objective settings, logging), plugin management (vis, neuroevolution, sibling projects, custom plugins), performance optimization (GPU parallelism, batch evaluation, `torch.compile`, population tuning, distributed deployment, profiling) |
| `tutorial_part5.md` | **5. Development and Extension** — Implementing custom `Problem` subclasses (`evaluate` with batch input), implementing custom `Algorithm` subclasses (full PSO example using `Mutable`/`Parameter`), custom monitors/operators, API tour (`evox.algorithms.so`, `evox.algorithms.mo`, `evox.problems`, `evox.metrics`), ML/RL integration patterns (HPO wrapping, Brax neuroevolution, `ParamsAndVector`) |
| `tutorial_part6.md` | **6. Troubleshooting and Optimization** — Common issues (import errors, GPU not used, OOM, convergence stagnation, poor results, backend conflicts, version mismatch) with solutions, debugging tips (small-scale testing, print statements, IDE breakpoints, profiling, logging), performance tuning guide (progressive scaling, hardware monitoring, parallelism, batch evaluation, algorithm selection) |
| `tutorial_part7.md` | **7. Practical Examples** — Three complete worked examples: (1) single-objective PSO on 10D Rastrigin, (2) multi-objective NSGA-II on a custom bi-objective problem with Pareto front visualization, (3) hyperparameter optimization of logistic regression on breast cancer data using CMA-ES |

## Integration Points

### Upstream (who links here)
- **`docs/source/index.md`** — Root documentation landing page includes `tutorial/index` in the "Getting Started" toctree (caption: "Getting Started") and as a grid-item-card
- **`docs/source/locale/zh_CN/tutorial/`** — Contains Chinese translations of all 7 tutorial parts plus a translated `index.md`; these are NOT in this directory but are structurally parallel

### Downstream
- This directory has no child subdirectories — it is a leaf node

## Constraints
- The tutorial is written in **MyST Markdown** (`.md` with `{toctree}`, `{note}`, `{tip}`, `{figure}`, `{math}` directives)
- Parts should be read sequentially (Part 1 → Part 7) — each builds on prior knowledge
- Chinese translations live under `locale/zh_CN/tutorial/` and are maintained separately (the machine translation script at `docs/machine_translate.py` explicitly skips tutorials since they are manually translated)
- Figures referenced in tutorials (e.g., `rastrigin_function.svg`, `example_nsga2_result.svg`) live in `docs/source/_static/`
- The `index.md` toctree must stay in sync with the actual tutorial files; any new part must be added to the toctree

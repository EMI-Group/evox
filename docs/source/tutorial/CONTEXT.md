# docs/source/tutorial/ — EvoX Getting-Started Tutorial Series

## Intent
This directory contains the **primary onboarding path** for EvoX users — a 7-part English tutorial series published as part of the Sphinx documentation. It is linked from the documentation landing page (`source/index.md`) via both a toctree and a prominent "Tutorials" grid card. The tutorial is intended to be read sequentially: each part builds on concepts introduced in prior chapters, taking a new user from zero knowledge to implementing custom algorithms and solving real-world optimization problems.

## API Surface

| File | Title | Summary |
|---|---|---|
| `index.md` | Tutorial landing page | Minimal toctree listing all 7 parts in order |
| `tutorial_part1.md` | 1. Introduction | What EvoX is, its five key features (modular architecture, distributed execution, functional programming, visualization, algorithm/problem libraries), and use cases (large-scale optimization, multi-objective, HPO, neuroevolution, research) |
| `tutorial_part2.md` | 2. Installation and Environment Setup | Python 3.10+ requirement, PyTorch installation (CUDA/ROCm/CPU variants), Windows (one-click `.bat` script + manual), Linux (pip + Docker/Podman for ROCm), and verification via `import evox` |
| `tutorial_part3.md` | 3. Basic Operations | First EvoX script: PSO + Ackley function. Introduces the **Algorithm–Problem–Workflow–Monitor** quartet, `StdWorkflow.init_step()` / `.step()`, `EvalMonitor` methods, algorithm methods, and device control (`.to(device)`) |
| `tutorial_part4.md` | 4. Advanced Features | Custom configuration (parameter tuning, operator replacement, multi-objective settings, logging), plugin management (vis, neuroevolution extras, sibling projects, custom plugins), and performance optimization (GPU parallelism, batch evaluation, `torch.compile`, population sizing, distributed deployment) |
| `tutorial_part5.md` | 5. Development and Extension | Custom `Problem` subclass (implement `evaluate` with batch support), custom `Algorithm` subclass (full PSO reimplementation using `Mutable`/`Parameter`), API tour (`evox.algorithms.so`/`.mo`, `evox.problems.numerical`/`.neuroevolution`/`.hpo_wrapper`, `evox.metrics`), ML and RL integration patterns |
| `tutorial_part6.md` | 6. Troubleshooting and Optimization | Common issues (import errors, GPU unused, OOM, convergence stagnation, JAX vs PyTorch, version mismatches), debugging techniques (small-scale testing, print/breakpoints, unit tests, profiling, file logging), and a performance tuning checklist |
| `tutorial_part7.md` | 7. Practical Examples | Three end-to-end examples: (1) PSO on 10D Rastrigin, (2) NSGA-II on a custom bi-objective problem with Pareto front plotting, (3) CMA-ES for HPO tuning logistic regression on the breast cancer dataset |

## Constraints
- **Sequential dependency**: Part 3 assumes Part 2 (installation); Part 5 assumes Part 3 (basic workflow); Part 7 demonstrates concepts from all prior parts. The toctree ordering should not be changed casually.
- **Chinese translations**: Every file here has a corresponding Chinese translation under `../locale/zh_CN/tutorial/` (mirrors the same 8-file structure). When adding, removing, or reordering tutorial pages, the Chinese translations must be updated accordingly. The translations are maintained manually (not machine-translated from the `.po` catalog — see `docs/machine_translate.py` which explicitly skips tutorial content).
- **MyST Markdown**: All files use MyST-flavored Markdown (Sphinx-compatible) — directives like `{toctree}`, `{tip}`, `{note}`, `{figure}`, `{math}`, and `{eval-rst}` may appear.
- **Static asset references**: Tutorial part 7 references static assets under `/_static/` (SVG figures of Rastrigin function and NSGA-II population plot).

## Routing Table
| Area | Path | Description |
|---|---|---|
| Tutorial landing page | `index.md` | Sphinx toctree — maps the 7 parts for sequential navigation |
| Part 1 – Introduction | `tutorial_part1.md` | EvoX overview, features, and use cases |
| Part 2 – Installation | `tutorial_part2.md` | Environment setup on Windows and Linux |
| Part 3 – Basic Operations | `tutorial_part3.md` | First script, core concepts, and basic API |
| Part 4 – Advanced Features | `tutorial_part4.md` | Configuration, plugins, and performance |
| Part 5 – Development | `tutorial_part5.md` | Custom modules, API reference, ML/RL integration |
| Part 6 – Troubleshooting | `tutorial_part6.md` | Common issues, debugging, and tuning |
| Part 7 – Practical Examples | `tutorial_part7.md` | Single/multi-objective and HPO walkthroughs |
| Chinese translations | `../locale/zh_CN/tutorial/` | Per-file zh_CN mirrors of the entire tutorial |

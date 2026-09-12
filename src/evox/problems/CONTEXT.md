# Problems

## Intent
Defines all problem/evaluation domains for EvoX. Every problem extends `evox.core.Problem` (from `evox.core.components`) which mandates an `evaluate(pop) -> Tensor` method returning fitness values for a population. Problems are the "evaluation function" in the evolutionary computation loop — given candidate solutions, they return scalar (single-objective) or vector (multi-objective) fitness.

## API Surface
- `evox.problems.numerical` → Classic numerical benchmark problems (single-objective, multi-objective, CEC suites)
- `evox.problems.neuroevolution` → Reinforcement learning / policy-search tasks (Brax, MuJoCo Playground, supervised learning)
- `evox.problems.hpo_wrapper` → Meta-optimization: `HPOProblemWrapper` wraps an entire `evox.core.Workflow` into a `Problem`, enabling hyperparameter optimization of algorithms

### Shipped user-facing Problem classes (inventory)
This node ships **22** concrete, instantiable `evox.core.Problem` subclasses (excluding abstract bases, monitors/data containers, aliases, and the legacy unexported class).
Counting the CEC2022 suite as its 12 individual functions instead of one wrapper class raises the "distinct benchmark functions" total to **33**.
| Group | Count | Classes (file) |
|---|---|---|
| Numerical — basic single-objective | 9 | Ackley, Griewank, Rastrigin, Rosenbrock, Schwefel, Sphere, Ellipsoid, Zakharov, Levy (`numerical/basic.py`) |
| Numerical — DTLZ multi-objective | 7 | DTLZ1–DTLZ7 (`numerical/dtlz.py`) |
| Numerical — CEC2022 | 1 | `CEC2022` (`numerical/cec2022.py`) — one class parameterized by `problem_number=1..12` |
| Neuroevolution / RL | 4 | `SupervisedLearningProblem` (`neuroevolution/supervised_learning.py`), `VirtualProblem` (`neuroevolution/virtual_problem.py`), `BraxProblem` (`neuroevolution/brax.py`), `MujocoProblem` (`neuroevolution/mujoco_playground.py`) |
| HPO | 1 | `HPOProblemWrapper` (`hpo_wrapper.py`) |
Excluded from the count: `ShiftAffineNumericalProblem` and `DTLZ` (abstract bases), `HPOMonitor` (ABC monitor base), `HPOFitnessMonitor` (concrete monitor, not a Problem), `HPOData` and `ModelStateForwardResult` (NamedTuple containers), and the legacy `VirtualLoRAProblem` class in `neuroevolution/virtual_lora_problem.py` (superseded, not exported).
Alias (not counted separately): `VirtualLoRAProblem = VirtualProblem` in `neuroevolution/virtual_problem.py`, re-exported by `neuroevolution/__init__.py`.
`BraxProblem` / `MujocoProblem` are single wrapper classes over the external Brax and MuJoCo-Playground registries (which list dozens of environments); the environments are not shipped in `src/`, so they count as 1 class each here.
Reproduce: `rg -n "^class " src/evox/problems` (29 class definitions total, of which the 22 above are user-facing Problem classes).
The published claims are **30+** benchmark problems/environments built into EvoX and **240+** across the EvoX ecosystem (README.md, README_ZH.md, and the docs site).
This node contributes the 22 user-facing Problem classes (≈33 distinct benchmark functions) inventoried above.
The remaining numerical suites (ZDT / MaF / LSMOP) and EvoXBench live in sibling EMI-Group libraries (e.g. EvoMO), not in `src/`.

## Constraints
- All problem classes MUST inherit from `evox.core.Problem` and implement `evaluate(self, pop: torch.Tensor) -> torch.Tensor`.
- Problems MUST be torch-compilable (`torch.compile`) and support `torch.func.vmap` for batched evaluation.
- Neuroevolution problems (`BraxProblem`, `MujocoProblem`) bridge Torch↔JAX via DLPack for GPU-accelerated physics simulation. They do **not** support `HPOProblemWrapper` out-of-box (vmap incompatibility); use `pop_size` workaround described in their docstrings.
- HPO-wrapped workflows require their monitor to be an `HPOMonitor` subclass (typically `HPOFitnessMonitor`).

## Routing Table
- `./numerical/` — Classic numerical benchmarks (Ackley, Rastrigin, Rosenbrock, etc.), DTLZ multi-objective suite, CEC 2022 test suite
- `./neuroevolution/` — Neuroevolution environments: Brax physics sims, MuJoCo Playground, supervised learning loss-landscape
- `./hpo_wrapper.py` — `HPOProblemWrapper`, `HPOMonitor`, `HPOFitnessMonitor`, `HPOData`

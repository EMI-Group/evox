# Problems

## Intent
Defines all problem/evaluation domains for EvoX. Every problem extends `evox.core.Problem` (from `evox.core.components`) which mandates an `evaluate(pop) -> Tensor` method returning fitness values for a population. Problems are the "evaluation function" in the evolutionary computation loop — given candidate solutions, they return scalar (single-objective) or vector (multi-objective) fitness.

## API Surface
- `evox.problems.numerical` → Classic numerical benchmark problems (single-objective, multi-objective, CEC suites)
- `evox.problems.neuroevolution` → Reinforcement learning / policy-search tasks (Brax, MuJoCo Playground, supervised learning)
- `evox.problems.hpo_wrapper` → Meta-optimization: `HPOProblemWrapper` wraps an entire `evox.core.Workflow` into a `Problem`, enabling hyperparameter optimization of algorithms

## Constraints
- All problem classes MUST inherit from `evox.core.Problem` and implement `evaluate(self, pop: torch.Tensor) -> torch.Tensor`.
- Problems MUST be torch-compilable (`torch.compile`) and support `torch.func.vmap` for batched evaluation.
- Neuroevolution problems (`BraxProblem`, `MujocoProblem`) bridge Torch↔JAX via DLPack for GPU-accelerated physics simulation. They do **not** support `HPOProblemWrapper` out-of-box (vmap incompatibility); use `pop_size` workaround described in their docstrings.
- HPO-wrapped workflows require their monitor to be an `HPOMonitor` subclass (typically `HPOFitnessMonitor`).

## Routing Table
- `./numerical/` — Classic numerical benchmarks (Ackley, Rastrigin, Rosenbrock, etc.), DTLZ multi-objective suite, CEC 2022 test suite
- `./neuroevolution/` — Neuroevolution environments: Brax physics sims, MuJoCo Playground, supervised learning loss-landscape
- `./hpo_wrapper.py` — `HPOProblemWrapper`, `HPOMonitor`, `HPOFitnessMonitor`, `HPOData`

# Numerical Benchmarks

## Intent
Classic numerical optimization benchmark problems for EvoX — the standard "test suite" used to evaluate and compare evolutionary algorithms. All problems inherit from `evox.core.Problem` (via intermediate base classes defined here) and implement `evaluate(pop) -> Tensor` returning fitness values.

Three families are provided:
- **Single-objective classical functions** (Ackley, Rastrigin, Rosenbrock, Sphere, etc.) — each available as both an `evox.core.Problem` subclass (for plugging into workflows) and a standalone pure `*_func` callable (for direct use).
- **DTLZ multi-objective suite** (DTLZ1–DTLZ7) — multi-objective problems with Pareto front sampling via `pf()`.
- **CEC 2022 single-objective test suite** — the 12 official CEC 2022 benchmark problems, loading pre-computed rotation matrices, shift vectors, and shuffle indices from bundled data files.

## API Surface — Exports

### Submodules (importable as `evox.problems.numerical.basic`, `.cec2022`, `.dtlz`)
| Module | Description |
|---|---|
| `basic` | `ShiftAffineNumericalProblem` base + 9 classical single-objective functions (Problem classes + pure functions) |
| `dtlz` | `DTLZ` base + DTLZ1–DTLZ7 multi-objective problems |
| `cec2022` | `CEC2022` — the 12 CEC 2022 single-objective problems |
| `cec2022_input_data/` | ~54 pre-computed `.txt` data files (rotation matrices `M_*.txt`, shift vectors `shift_data_*.txt`, shuffle indices `shuffle_data_*.txt`) |

### Concrete Problem classes (defined in this package)
| Class | Parent | Type | Minima |
|---|---|---|---|
| `Ackley` | `ShiftAffineNumericalProblem` | Single-objective | x = [0, …, 0] |
| `Griewank` | `ShiftAffineNumericalProblem` | Single-objective | x = [0, …, 0] |
| `Rastrigin` | `ShiftAffineNumericalProblem` | Single-objective | x = [0, …, 0] |
| `Rosenbrock` | `ShiftAffineNumericalProblem` | Single-objective | x = [1, …, 1] |
| `Schwefel` | `ShiftAffineNumericalProblem` | Single-objective | x = [420.9687, …] |
| `Sphere` | `ShiftAffineNumericalProblem` | Single-objective | x = [0, …, 0] |
| `Ellipsoid` | `ShiftAffineNumericalProblem` | Single-objective | x = [0, …, 0] |
| `Zakharov` | `ShiftAffineNumericalProblem` | Single-objective | x = [0, …, 0] |
| `Levy` | `ShiftAffineNumericalProblem` | Single-objective | x = [1, …, 1] |
| `DTLZ1`–`DTLZ7` | `DTLZ` | Multi-objective | Pareto front via `pf()` |
| `CEC2022` | `Problem` | Single-objective | 12 functions, shifted/rotated |

Re-export nuance: `numerical/__init__.py` re-exports only 7 of the 9 basic classes (`Ackley, Griewank, Rastrigin, Rosenbrock, Schwefel, Sphere, Ellipsoid`).
`Zakharov` and `Levy` are declared in `basic.__all__` but are NOT re-imported by `numerical/__init__.py`, so they are reachable only via `evox.problems.numerical.basic.Zakharov` / `.Levy`.

### Pure functions (no shift/affine wrapping)
`ackley_func`, `griewank_func`, `rastrigin_func`, `rosenbrock_func`, `schwefel_func`, `sphere_func`, `ellipsoid_func`, `zakharov_func`, `levy_func`
`zakharov_func` and `levy_func` are likewise reachable only via `basic`, not re-exported at the `numerical` level.

## Class Hierarchy & Design

```
evox.core.Problem (ABC)
├── ShiftAffineNumericalProblem           # basic.py — shift + affine transform pipeline
│   ├── Ackley, Griewank, Rastrigin, Rosenbrock, Schwefel, Sphere, Ellipsoid, Zakharov, Levy
├── DTLZ                                  # dtlz.py — multi-objective base, Pareto front sampling
│   ├── DTLZ1
│   ├── DTLZ2 ─── DTLZ3, DTLZ4            # DTLZ3/4 reuse DTLZ2's shape-building logic
│   ├── DTLZ5, DTLZ6, DTLZ7
├── CEC2022                               # cec2022.py — loads data, exposes f1–f12
```

### `ShiftAffineNumericalProblem` (basic.py)
Applies optional shift vector and affine (rotation/scaling) matrix to input population before delegating to `_true_evaluate()`. This decouples the raw function from spatial transformations, allowing the same function to be used shifted/rotated without modifying its implementation.

- Constructor: `__init__(self, shift=None, affine=None)` — both optional, each is a `torch.Tensor`.
- `evaluate(pop)`: applies `pop + shift` then `pop @ affine`, then calls `self._true_evaluate(pop)`.
- Subclasses override only `_true_evaluate`, which delegates to the corresponding `*_func`.

### `DTLZ` (dtlz.py)
Base for multi-objective DTLZ problems. Uses uniform reference-point sampling in the constructor for Pareto front generation.

- Constructor: `__init__(self, d, m, ref_num=1000)` — decision variables `d`, objectives `m`, reference points `ref_num`.
- `evaluate(X) -> Tensor` shape `(n, m)`: returns `m` objective values per candidate.
- `pf() -> Tensor`: returns the Pareto front approximation as sampled points.
- DTLZ3/4 inherit from DTLZ2 (not DTLZ directly) — they reuse DTLZ2's shape-building expressions with modified `g` functions.
- DTLZ7 uses `grid_sampling` instead of `uniform_sampling`.

### `CEC2022` (cec2022.py)
Implements the full CEC 2022 single-objective test suite (12 problems). On construction, loads rotation matrices (`M`), shift vectors (`OShift`), and shuffle indices (`SS`) from `cec2022_input_data/` text files.

- Constructor: `__init__(self, problem_number, dimension, device=None)`. Dimension must be 2, 10, or 20. Problems 6–8 are undefined for D=2.
- `evaluate(pop)`: dispatches to `cec2022_f1` through `cec2022_f12`.
- Problem categories:
  - **Basic shifted/rotated** (F1–F5): Zakharov, Rosenbrock, Schaffer F7, Step Rastrigin, Levy — each wrapped with `sr_func_rate`.
  - **Hybrid** (F6–F8): Partitioned input with different sub-functions per partition, then summed.
  - **Composition** (F9–F12): Weighted combination of multiple sub-functions via `cf_cal`.
- Contains private helper functions (`shift`, `rotate`, `sr_func_rate`, `cut`, `cf_cal`) and raw function implementations (`bent_cigar_func`, `hgbat_func`, `katsuura_func`, `modified_schwefel_func`, `schaffer_F7_func`, `escaffer6_func`, `happycat_func`, `grie_rosen_func`, `discus_func`, `ellips_func`, `levy_func`).

## Constraints
- All Problem classes MUST inherit from `evox.core.Problem` (transitively via the base classes here).
- All functions implement `evaluate(pop) -> Tensor` where pop has shape `(population_size, dimension)`.
- All pure `*_func` callables expect input of shape `(n, d)` and return shape `(n,)`.
- CEC2022 requires `cec2022_input_data/` to be present alongside `cec2022.py` — data files are loaded relative to `__file__`.
- DTLZ `pf()` methods may be overridden to provide exact Pareto fronts where the uniform-sampling default is insufficient.

## When to Use Classes vs. Pure Functions
- Use **Problem classes** (`Ackley(...)`) when plugging into an EvoX `Workflow` — they implement the full `Problem` interface and support shift/affine transformations.
- Use **pure functions** (`ackley_func(...)`) for direct evaluation without the transformation overhead, or as building blocks in composite problems (as `cec2022.py` does, importing `ackley_func`, `rastrigin_func`, etc. from `basic`).

## Routing Table
- `./basic.py` — `ShiftAffineNumericalProblem` base class + 9 single-objective classical functions (Problem classes + pure functions)
- `./dtlz.py` — `DTLZ` base class + DTLZ1–DTLZ7 multi-objective problems
- `./cec2022.py` — `CEC2022` class (12 CEC 2022 single-objective problems)
- `./cec2022_input_data/` — Pre-computed rotation, shift, and shuffle data files for CEC 2022
- `./__init__.py` — Re-exports all concrete classes and pure functions; imports submodules

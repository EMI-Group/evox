# `src/evox/algorithms/so/pso_variants/` — Particle Swarm Optimization Variants

## Intent
This directory contains Particle Swarm Optimization (PSO) algorithm variants for single-objective optimization. All are subclasses of `evox.core.Algorithm` and follow the standard tensorized, GPU-accelerated design pattern.

## Common PSO Pattern (Velocity-Position Update)

All PSO variants share a core iterative loop driven by velocity-position updates on **bounded continuous spaces** (`lb`/`ub` tensors):

1. Evaluate current population fitness (`self.evaluate(self.pop)`)
2. Update personal/local/global best locations and fitness values
3. Compute velocity update as a weighted combination of:
   - **Inertia**: current velocity scaled by `w` (or random coefficient)
   - **Social learning**: movement toward guiding exemplars (global best, personal best, demonstrator, etc.)
4. Update position: `pop = pop + velocity`
5. Clamp both position and velocity to `[lb, ub]`

The key differentiator across variants is **how social learning is structured** — i.e., which exemplar(s) each particle learns from.

## Shared Utility (`utils.py`)

| Function | Purpose |
|---|---|
| `min_by(values, keys)` | Finds the value with the minimum key across concatenated tensors. Used to track the global best. Takes `List[Tensor]` for both values and keys, concatenates along dim 0, and returns `(best_value, best_key)`. |

## Variant-by-Variant Breakdown

### 1. `PSO` (`pso.py`) — Classic Particle Swarm Optimization
The canonical PSO: each particle maintains a personal best and the swarm shares a single global best.

**Velocity update:**
```
v = w * v + phi_p * rp * (pbest - x) + phi_g * rg * (gbest - x)
```

**State (`Mutable`):** `pop`, `velocity`, `fit`, `local_best_location`, `local_best_fit`, `global_best_location`, `global_best_fit`

**Parameters:** `w` (inertia, 0.6), `phi_p` (cognitive, 2.5), `phi_g` (social, 0.8)

### 2. `CLPSO` (`clpso.py`) — Comprehensive Learning PSO
Instead of learning from its own personal best, each particle **learns from other particles' personal bests**. A tournament selects between two random particles' personal bests, and with probability `P_c` (learning probability), a particle uses the winner's personal best as its exemplar (otherwise uses its own).

**Key difference from PSO:** No global best term in velocity update. The exemplar `personal_best` is chosen per-dimension via tournament from two randomly selected particles, with `P_c` probability of cross-learning.

**Velocity update:**
```
v = w * v + c * rand * (chosen_pbest - x)
```

**Parameters:** `w` (inertia, 0.5), `c` (coefficient, 1.5), `P_c` (learning probability, 0.05)

### 3. `CSO` (`cso.py`) — Competitive Swarm Optimizer
A generational variant: particles are randomly paired into **competitions**, where the winner (better fitness) becomes the **teacher** and the loser becomes the **student**. Only students update; teachers are preserved (elitism). Students learn from their paired teacher plus a convergence term toward the population center.

**Velocity update (student only):**
```
v_student = lambda1 * v_student + lambda2 * (x_teacher - x_student) + phi * lambda3 * (center - x_student)
```

**State:** `pop`, `velocity`, `fit` (no personal/global best tracking — the competition mechanism replaces explicit bests)

**Parameters:** `phi` (center convergence weight, 0.0)

### 4. `DMSPSOEL` (`dms_pso_el.py`) — Dynamic Multi-Swarm PSO with Ensemble Learning
The most complex variant. The population is split into two groups:
- **Dynamic sub-swarms** (first N particles): Each sub-swarm has its own local best. Periodically regrouped (shuffled) every `regrouped_iteration_num` steps.
- **Following sub-swarm** (remaining particles): Learns from **regional bests** — the best particles from the dynamic swarms.

Has a two-phase strategy switch at `0.9 * max_iteration`:
- **Strategy 1** (exploration): Dynamic swarms learn from pbest + lbest; following swarm learns from pbest + rbest (regional best)
- **Strategy 2** (exploitation, late-stage): All particles learn from pbest + gbest (classic PSO style)

**State:** `pop`, `velocity`, `fit`, `personal_best_location`, `personal_best_fit`, `local_best_location`, `local_best_fit`, `regional_best_index`, `global_best_location`, `global_best_fit`, `iteration`

**Parameters:** `dynamic_sub_swarm_size` (10), `dynamic_sub_swarms_num` (5), `following_sub_swarm_size` (10), `regrouped_iteration_num` (50), `max_iteration` (100), `w` (0.7), `c_pbest` (1.5), `c_lbest` (1.5), `c_rbest` (1.0), `c_gbest` (1.0)

### 5. `FSPSO` (`fs_pso.py`) — Fitness-based PSO (Feature Selection PSO)
An elitism-based variant. The population is sorted by fitness; the **top half** (elite) performs a standard PSO velocity update (pbest + gbest), while the **bottom half** is replaced via **tournament selection + mutation** from the elite pool.

**Key differences from PSO:**
- Only the elite half contributes to personal/global best updates
- Offspring are generated by copying elite parents (tournament winners) and applying random perturbation with probability `mutate_rate`
- Combines PSO velocity update with genetic-style mutation

**Parameters:** `w` (0.6), `phi_p` (2.5), `phi_g` (0.8), `mutate_rate` (0.01), plus optional `mean`/`stdev` for Gaussian initialization

### 6. `SLPSOGS` (`sl_pso_gs.py`) — Social Learning PSO with Gaussian Sampling
No personal best tracking. Instead, particles learn from **demonstrators** — better-performing particles chosen via a **Gaussian sampling** mechanism. Particles are sorted worst-to-best; each particle selects a demonstrator from the better-ranked portion using a normal distribution whose standard deviation is scaled by `demonstrator_choice_factor`. Also includes a convergence term toward the population mean.

**Velocity update:**
```
v = r1 * v + r2 * (X_k - x) + r3 * epsilon * (X_avg - x)
```
where `X_k` is the Gaussian-sampled demonstrator.

**Parameters:** `social_influence_factor` (epsilon, 0.2), `demonstrator_choice_factor` (theta, 0.7)

### 7. `SLPSOUS` (`sl_pso_us.py`) — Social Learning PSO with Uniform Sampling
Identical structure to SLPSOGS, but demonstrators are chosen via **uniform sampling** rather than Gaussian. Particles sort worst-to-best; each particle's demonstrator is uniformly sampled from a range `[q, pop_size)` where `q` depends on the particle's rank and `demonstrator_choice_factor`.

**Velocity update:** Same form as SLPSOGS:
```
v = r1 * v + r2 * (X_k - x) + r3 * epsilon * (X_avg - x)
```

**Parameters:** `social_influence_factor` (epsilon, 0.2), `demonstrator_choice_factor` (theta, 0.7)

## Taxonomy by Learning Mechanism

| Mechanism | Variants |
|---|---|
| **Global best (gbest)** — single best across the swarm | PSO, FSPSO (elite only), DMSPSOEL (strategy 2) |
| **Personal best (pbest)** — each particle's own history | PSO, DMSPSOEL, FSPSO (elite only) |
| **Cross-learning from other pbests** | CLPSO |
| **Local best (lbest)** — best within a sub-swarm | DMSPSOEL (strategy 1) |
| **Regional best (rbest)** — best individuals from other sub-swarms | DMSPSOEL (strategy 1) |
| **Pairwise competition (teacher-student)** | CSO |
| **Demonstrator selection (ranked sampling)** | SLPSOGS, SLPSOUS |
| **Tournament + mutation (hybrid GA)** | FSPSO |
| **Population mean convergence** | CSO, SLPSOGS, SLPSOUS |

## API Surface

| Module | Class | Inherits From |
|---|---|---|
| `pso.py` | `PSO` | `evox.core.Algorithm` |
| `clpso.py` | `CLPSO` | `evox.core.Algorithm` |
| `cso.py` | `CSO` | `evox.core.Algorithm` |
| `dms_pso_el.py` | `DMSPSOEL` | `evox.core.Algorithm` |
| `fs_pso.py` | `FSPSO` | `evox.core.Algorithm` |
| `sl_pso_gs.py` | `SLPSOGS` | `evox.core.Algorithm` |
| `sl_pso_us.py` | `SLPSOUS` | `evox.core.Algorithm` |
| `utils.py` | `min_by()` | (standalone function) |
| `__init__.py` | Re-exports all 7 classes | — |

## Dependencies

- **Internal:** `evox.core` (`Algorithm`, `Mutable`, `Parameter`), `evox.utils` (`clamp`, `clamp_int`), `.utils` (`min_by`)
- **External:** `torch` only — all operations are pure PyTorch, GPU-compatible

## Constraints

- All algorithms operate on **bounded continuous spaces** with 1D `lb`/`ub` tensors of equal shape and dtype.
- State tensors must use `Mutable`; hyperparameters must use `Parameter`.
- `init_step()` performs first evaluation and initial bookkeeping; `step()` performs one generation.
- `self.evaluate(pop)` is injected by the workflow (proxy to `Problem.evaluate`) — never defined here.
- Velocity and position are always clamped to `[lb, ub]` after updates.
- Population initialization: uniform random in `[lb, ub]`; velocity initialized as `2 * range * U(0,1) - range`.

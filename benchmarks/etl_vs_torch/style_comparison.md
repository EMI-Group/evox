# Style comparison: torch OOP evox vs functional evox_etl

This report compares the **code style and cleanliness** of the two
implementations exercised by this benchmark: the PyTorch-native OOP `evox`
(`src/evox/`, `torch.nn.Module`-based `Algorithm` classes with in-place
`Mutable` state) and its functional port `evox_etl` (`src/evox_etl/`, frozen
config/state dataclasses plus pure `init`/`ask`/`tell` functions traced by the
`etl` tensor library). Both implement the same algorithms/problems with 1:1
semantics (PSO, DE, CMA-ES, OpenES, NSGA2/NSGA3/MOEAD on
Sphere/Rastrigin/Ackley/DTLZ1/DTLZ2), so this is purely about how each style
expresses the same computation — readability, state handling, device/dtype/
randomness explicitness, and reasoning about compiled/traced behavior.
Measured numbers live in `results/*.json` and the `BENCHMARK_RESULTS.md`
ms/step summary generated alongside them.

## 1. Initialization: object constructor vs `init(config, key) -> state`

Torch — state is allocated and mutated into the object at construction
(`src/evox/algorithms/so/pso_variants/pso.py`):

```python
class PSO(Algorithm):
    def __init__(self, pop_size, lb, ub, w=0.6, phi_p=2.5, phi_g=0.8, device=None):
        super().__init__()
        device = torch.get_default_device() if device is None else device
        # Parameter marks hyper-parameters so they can be traced/vector-mapped
        self.w = Parameter(w, device=device)
        self.phi_p = Parameter(phi_p, device=device)
        self.phi_g = Parameter(phi_g, device=device)
        lb = lb[None, :].to(device=device)
        ub = ub[None, :].to(device=device)
        length = ub - lb
        pop = torch.rand(self.pop_size, self.dim, device=device)   # global RNG
        pop = length * pop + lb
        ...
        # mutable state lives on self, replaced in-place each step
        self.pop = Mutable(pop)
        self.velocity = Mutable(velocity)
        self.fit = Mutable(torch.full((self.pop_size,), torch.inf, device=device))
        ...
        self.global_best_fit = Mutable(torch.tensor(torch.inf, device=device))
```

etl — a frozen config, a frozen state, and a pure function between them
(`src/evox_etl/algorithms/so/pso_variants/pso.py`):

```python
@dataclass(frozen=True)
class PSO:
    pop_size: int
    lb: np.ndarray
    ub: np.ndarray
    w: float = 0.6
    phi_p: float = 2.5
    phi_g: float = 0.8

@dataclass(frozen=True)
class PSOState:
    pop: Tensor
    velocity: Tensor
    fit: Tensor
    ...
    global_best_fit: Tensor
    key: Tensor            # PRNG state travels with the algorithm state

def init(config: PSO, key: Tensor) -> PSOState:
    key, subkey = random.split(key)
    lb, ub = _bounds(config)
    length = ub - lb
    pop_size, dim = config.pop_size, len(config.lb)
    subkey_pop, subkey_vel = random.split(subkey)
    pop = length * random.uniform(subkey_pop, (pop_size, dim), 0.0, 1.0, etl.float32) + lb
    ...
    inf_fit = enp.full((pop_size,), float("inf"), dtype=etl.float32)
    return PSOState(pop=pop, velocity=velocity, fit=inf_fit,
                    local_best_location=pop, local_best_fit=inf_fit, ..., key=key)
```

## 2. One generation: in-place `step()` vs pure `ask`/`tell`

Torch — one method mutates `self`; `self.evaluate` is a proxy injected by
`StdWorkflow` (`src/evox/algorithms/so/pso_variants/pso.py`, `step`):

```python
    def step(self):
        compare = self.local_best_fit > self.fit
        self.local_best_location = torch.where(compare[:, None], self.pop, self.local_best_location)
        self.local_best_fit = torch.where(compare, self.fit, self.local_best_fit)
        self.global_best_location, self.global_best_fit = min_by(
            [self.global_best_location.unsqueeze(0), self.pop],
            [self.global_best_fit.unsqueeze(0), self.fit],
        )
        rg = torch.rand(self.pop_size, self.dim, device=self.fit.device)   # global RNG
        rp = torch.rand(self.pop_size, self.dim, device=self.fit.device)
        velocity = (
            self.w * self.velocity
            + self.phi_p * rp * (self.local_best_location - self.pop)
            + self.phi_g * rg * (self.global_best_location - self.pop)
        )
        pop = self.pop + velocity
        self.pop = clamp(pop, self.lb, self.ub)
        self.velocity = clamp(velocity, self.lb, self.ub)
        self.fit = self.evaluate(self.pop)
```

etl — `ask` returns `(candidates, new_state)`, `tell` folds the fitness back
in; each is a pure function of `(config, state)`
(`src/evox_etl/algorithms/so/pso_variants/pso.py`):

```python
def ask(config: PSO, state: PSOState) -> Tuple[Tensor, PSOState]:
    lb, ub = _bounds(config)
    pop_size, dim = config.pop_size, len(config.lb)
    compare = state.local_best_fit > state.fit
    local_best_location = etl.select(
        enp.expand_dims(compare, axis=1), state.pop, state.local_best_location
    )
    local_best_fit = etl.select(compare, state.fit, state.local_best_fit)
    global_best_location, global_best_fit = min_by(...)    # same math as torch
    key, subkey = random.split(state.key)          # PRNG threaded explicitly
    subkey_rg, subkey_rp = random.split(subkey)
    rg = random.uniform(subkey_rg, (pop_size, dim), 0.0, 1.0, etl.float32)
    rp = random.uniform(subkey_rp, (pop_size, dim), 0.0, 1.0, etl.float32)
    velocity = (
        config.w * state.velocity
        + config.phi_p * rp * (local_best_location - state.pop)
        + config.phi_g * rg * (global_best_location - state.pop)
    )
    pop = clamp(state.pop + velocity, lb, ub)
    velocity = clamp(velocity, lb, ub)
    return pop, PSOState(pop=pop, velocity=velocity, fit=state.fit, ..., key=key)

def tell(config: PSO, state: PSOState, fitness: Tensor) -> PSOState:
    """Record the evaluated fitness (torch ``step`` after ``evaluate``)."""
    return PSOState(pop=state.pop, velocity=state.velocity, fit=fitness, ..., key=state.key)
```

## 3. Workflow: hidden in-place state vs explicit state threading

Torch — construct from live objects; the workflow moves everything to the
device **in-place** and injects `evaluate` by building a `_SubAlgorithm` class
at runtime (`src/evox/workflows/std_workflow.py`; driver from
`benchmarks/etl_vs_torch/bench_so.py`):

```python
algo = PSO(case.pop_size, lb, ub, device=device)
problem = Sphere()
monitor = EvalMonitor(full_fit_history=True)
workflow = StdWorkflow(algorithm=algo, problem=problem,
                       monitor=monitor, opt_direction="min", device=device)
workflow.init_step()
for _ in range(case.gens):
    workflow.step()          # state lives in module attributes; nothing returned
```

etl — construct from frozen configs; `init(seed=...)` builds + compiles the step
graph once and returns the whole workflow state; every step is `state in,
state out` (`src/evox_etl/core/workflow.py`; driver from
`benchmarks/etl_vs_torch/bench_so.py`):

```python
workflow = StdWorkflow(algorithm=PSO(pop_size=case.pop_size, lb=lb, ub=ub),
                       problem=Sphere(),
                       monitor=EvalMonitorConfig(pop_size=case.pop_size, dim=case.dim),
                       opt_direction="min", backend=etl_backend, device=device)
state = workflow.init(seed=case.seed)   # graph build + backend compile
for _ in range(case.gens):
    state = workflow.step(state)        # pure: state in, state out
```

## 4. What each style buys you

### torch OOP (`src/evox/`)

**Pros**

- **Familiar and introspectable**: plain `torch.nn.Module` — anyone who knows
  PyTorch can read `step()`; named submodules/`Parameter`/`Mutable` attributes
  make state easy to inspect interactively, and `StdWorkflow`'s in-place
  `.to(device)` (`std_workflow.py:121-123`) matches `nn.Module` habits.
- **Terse, dynamic, instant**: mutation is one assignment (`self.fit =
  self.evaluate(self.pop)`), shapes are unconstrained, and the benchmark runs
  torch eagerly (`compile_time_s = 0.0`) — edit-run-debug loops are instant.

**Cons**

- **Implicit device/dtype**: `nn.Module.to()` only moves parameters and
  buffers, not plain-attribute tensors. Hyperparameters stored as plain
  attributes (`self.lb`, `self.ub` in PSO) keep whatever device they were
  built on, and several MO algorithms allocate without any device argument —
  `bench_mo.py:101-118` had to `torch.set_default_device(device)` because
  NSGA2/NSGA3/MOEAD population init uses the raw constructor `lb`/`ub` and
  NSGA3's `torch.arange`/`uniform_sampling` land on the default device.
- **Hidden state**: RNG is implicit — `torch.rand` reads the global generator
  and seeding happens out-of-band (`torch.manual_seed`), so reproducibility
  depends on call order of unrelated code.
- **Hard to reason about compiled behavior**: `evox.core.compile`/`vmap` need
  scalar-index graph-break workarounds, monitors run outside the compiled
  graph, and `StdWorkflow.__init__` synthesizes a `_SubAlgorithm` subclass
  (`std_workflow.py:129-135`) to inject `evaluate` — exactly the
  metaprogramming a tracer dislikes.
- **Silent NaN propagation**: the CMA-ES `c_c` formula
  (`src/evox/algorithms/so/es_variants/cma_es.py:67-68`) exceeds 2 at
  pop≥1000/dim≥50, making `torch.sqrt(self.c_c * (2 - self.c_c) *
  self.mu_eff)` (`cma_es.py:134-136`) NaN **inside the graph**. On CUDA,
  cusolver eventually raises `_LinAlgError`; on CPU, LAPACK silently
  propagates the NaN covariance, so the torch-cpu CMA-ES numbers in
  `results/so_torch-cpu.json` are silently wrong (see the `note` fields in
  `results/so_torch-cuda.json`).
### etl functional (`src/evox_etl/`)

**Pros**

- **Explicit state threading**: `init(config, key) -> state`, and `ask`/`tell`
  are pure functions of `(config, state)` — same inputs, same outputs, every
  time. The whole workflow state (algorithm + problem + monitor + generation +
  PRNG key) is one returned `WorkflowState` (`core/workflow.py:24-37`).
- **Explicit randomness**: PRNG keys are fields of the state dataclass, split
  per use (`random.split(state.key)`), so a run is reproducible from the
  single `init(seed=...)` call.
- **Explicit placement**: backend and device are workflow arguments; device
  placement is one visible `tree_map(lambda t: t.to(device), state)`
  (`core/workflow.py:229-230`), and the graph is compiled once at `init()`.
- **Loud failure modes**: hyperparameters are host-side Python scalars
  validated at construction, so `math.sqrt(c_c * (2 - c_c) * mu_eff)`
  (`src/evox_etl/algorithms/so/es_variants/cma_es.py:234-235`) raises
  `ValueError: math domain error` immediately instead of NaN-ing the
  covariance (see the `note` in `results/so_etl-iree-cuda.json`), and compile
  errors point at exact source locations (stablehlo gap below).

**Cons**

- **Less familiar, more verbose**: every algorithm needs a frozen state
  dataclass, and `ask`/`tell` must rebuild the full state explicitly
  (`pso.py:154-163, 166-177`) where torch assigns one attribute.
- **Static shapes, no eager mode**: the tracer rejects numpy arrays/scalars as
  static trace arguments (`TraceError`), so bounds are stored as float tuples
  (`PSO.__post_init__`, `pso.py:43-51`), and everything runs through
  `etl.build`/`etl.run` — no dynamic shapes, and you cannot step through one
  tensor op interactively as in torch.
- **Op-coverage gaps surface at compile time**: the stablehlo exporter defers
  `cumprod`/`flip` used by DTLZ
  (`src/evox_etl/problems/numerical/dtlz.py:137,164`), so all 12 MO cases fail
  with `BackendError` on iree/xla even though the numpy backend runs the
  identical code fine (`results/mo_etl-iree-cuda.json`,
  `results/mo_etl-xla-cuda.json`).

## 5. Verdict

The two styles win in different phases of the same project. **Torch OOP wins
for prototyping and interactive use**: familiar `nn.Module` ergonomics,
dynamic shapes, zero compile wall, and easy introspective debugging make it
the right substrate for experimenting with new algorithms — exactly what
upstream `evox` targets. The cost is paid later: implicit device placement
(the MO harness needed `torch.set_default_device`), ambient randomness, and
errors like the CMA-ES NaN covariance propagating silently through an entire
CPU benchmark run.

**etl functional wins for deployable compiled/GPU execution and
reproducibility**: pure functions over explicit state, keys stored in the
state, loud host-side failures, and one compiled graph per run. The current
adapter reality is visible in the numbers: `results/so_etl-xla-cuda.json`
shows PSO/Sphere at ~5.4 ms/step (100x10) and ~17 ms/step (10000x100) after
~2-4 s of compile, while `results/so_torch-cuda.json` holds ~1.1 ms/step at
every scale in eager mode — but the etl staging cost is adapter overhead
(host↔device copies per `etl.run`, see `results/README.md`), not the
functional style itself, and the MO gaps are compiler-adapter gaps, not
language gaps. Per-case tables, parity checks, and error notes live in
`results/*.json` (with the `BENCHMARK_RESULTS.md` ms/step summary generated
alongside them).

**Bottom line**: write the algorithm in torch OOP style, and port it to the
functional style when it must run reproducibly and loudly at scale — the two
are views of the same math, and this benchmark exists to verify that they
agree.

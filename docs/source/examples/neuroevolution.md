# Neuroevolution: the training-free virtual ES family

This page trains a small neural network without gradients: `VirtualES` and `VirtualLoRAES` treat the
network's flattened parameters as a search vector and evaluate a **virtual population** of perturbed
networks — no `(pop_size, dim)` matrix of candidates is ever built.
The problem that scores those perturbations is `VirtualProblem`, which regenerates the Gaussian noise
from per-individual seeds inside the compiled graph.

```{warning} Experimental EvoX 2.0 pre-release
The code below uses the `evox_etl` import name (the functional rewrite on the `evoxc` compiler) and
the **NumPy backend** (`backend="numpy"`). The 1.x Brax / MuJoCo / supervised-learning problem
classes are **not** part of 2.0 — see the first section for what replaces them.
```

## What happened to the Brax, MuJoCo and supervised-learning problems

The 1.x `evox.problems.neuroevolution` package wrapped three external ecosystems: Brax and
MuJoCo-Playground (JAX environments) and supervised learning (a `torch.nn` model plus a
`DataLoader`). None of that survives the move to `evoxc`, because every evaluated function must be
traced into an `evoxc` graph, and because a `(pop_size, dim)` population of full networks is exactly
the allocation the virtual family was invented to avoid:

| 1.x problem | 1.x dependency | What you use in 2.0 |
|---|---|---|
| `BraxProblem` / `MujocoPlaygroundProblem` | JAX, Brax, MuJoCo-Playground | Nothing built in — a policy on a JAX physics simulator has no `evoxc` counterpart. Write the rollout as your own `Problem` config (see {doc}`../guide/developer/custom-alg-pro`) and drive it with the same `VirtualES`. |
| `SupervisedLearningProblem` | `torch.nn`, `DataLoader` | `make_virtual_problem(..., loss="mse")` or `loss="cross_entropy"`, with a fixed batch of inputs/targets flattened host-side. This is the page below. |
| `ParamsAndVector` (torch param ↔ flat-vector bridge) | `torch.nn.Module` | Not needed: the network *is* `param_shapes` + `layer_specs`, and its parameters *are* the algorithm's `(dim,)` center. |

So "neuroevolution" in 2.0 means: a fixed dataset or rollout loss, expressed as a `VirtualProblem`
over a flat parameter vector, optimized by a training-free ES that never materializes the
population.

## The network as data

Two tuples describe the network, in the order torch's `named_parameters()` would report them:

| Field | Type | Meaning |
|---|---|---|
| `param_shapes` | tuple of int tuples | Weight/bias shapes, e.g. `((8, 4), (8,), (1, 8), (1,))` for a 4 → 8 → 1 MLP |
| `layer_specs` | tuple of `("linear", weight_idx, bias_idx, activation)` | Which parameter index feeds each layer and what activation follows it (`bias_idx=None` for no bias) |

Here `dim = 8·4 + 8 + 1·8 + 1 = 49`, and that single flat vector of 49 numbers is the whole search
space: `VirtualES` stores a `center` of that length and nothing else.

Supported activations are `"relu"`, `"tanh"`, `"sigmoid"`, `"gelu"` and `"identity"`; supported
losses are `"mse"` and `"cross_entropy"`; the per-sample loss is reduced over the batch with
`reduction="mean"` or `"sum"`.

## The complete program

```python
import numpy as np
import evoxc
from evox_etl import StdWorkflow, EvalMonitorConfig
from evox_etl.algorithms import make_virtual_es
from evox_etl.problems.neuroevolution import evaluate, make_virtual_problem
from evox_etl.problems.numerical.state import ProblemState

# 1. A tiny fixed regression task: learn y = x @ w_true with a 4 -> 8 -> 1 MLP.
batch, in_features, out_features = 8, 4, 1
rng = np.random.default_rng(0)
inputs = rng.uniform(-1.0, 1.0, size=(batch, in_features)).astype(np.float32)
w_true = np.array([1.0, -0.5, 0.25, 0.75], dtype=np.float32)
targets = (inputs @ w_true).reshape(batch, out_features).astype(np.float32)

# 2. The network as (parameter shapes, layer specs) — torch named_parameters() order.
param_shapes = ((8, 4), (8,), (1, 8), (1,))
layer_specs = (("linear", 0, 1, "relu"), ("linear", 2, 3, "identity"))
dim = sum(int(np.prod(shape)) for shape in param_shapes)          # 49

problem = make_virtual_problem(
    param_shapes=param_shapes,
    layer_specs=layer_specs,
    inputs=inputs.reshape(-1).tolist(),          # one baked batch, flattened host-side
    targets=targets.reshape(-1).tolist(),
    in_features=in_features,
    out_features=out_features,
    batch_size=batch,
    loss="mse",
    reduction="mean",
)

# 3. The training-free ES: population size, step size, noise scale.
algorithm = make_virtual_es(
    param_shapes,
    pop_size=256,
    center_init=(0.0,) * dim,
    learning_rate=0.1,
    noise_stdev=0.1,
)

workflow = StdWorkflow(
    algorithm, problem,
    monitor=EvalMonitorConfig(),
    opt_direction="min",
    num_generations=100,
    backend="numpy",
)
state = workflow.run(seed=42)
print(f"best noisy fitness: {float(state.algorithm_state.best_fitness.numpy()):.6f}")
print(f"monitor best      : {workflow.monitor.get_best_fitness():.6f}")
```

```text
best noisy fitness: 0.000657
monitor best      : 0.000657
```

Both numbers refer to **perturbed** networks: the fitness of individual *i* is the loss of
`center + noise_stdev · Δ(seed_i)`, so a good "best fitness" alone does not prove the network you
finally deploy is good. Deploying it means evaluating the plain center network — noise scale `0`:

```python
def center_loss(config, center):
    """Loss of the unperturbed center network (perturbation scale pinned to 0)."""
    center_flat = np.asarray(center, dtype=np.float32).reshape(dim)

    def body(center_flat, seeds):
        fitness, _ = evaluate(config, ProblemState(), (center_flat, seeds, 0.0))
        return fitness

    exe = evoxc.build(
        body,
        evoxc.core.TensorSpec((dim,), np.float32),
        evoxc.core.TensorSpec((1,), np.int64),
        backend="numpy",
    )
    return float(np.mean(evoxc.run(exe, center_flat, np.zeros(1, np.int64)).numpy()))

print(f"center loss before: {center_loss(problem, np.zeros(dim, np.float32)):.6f}")
print(f"center loss after : {center_loss(problem, state.algorithm_state.center.numpy()):.6f}")
```

```text
center loss before: 0.519039
center loss after : 0.000718
```

The zero-initialized network starts at the loss of "predict 0 everywhere" (`0.519`) and the trained
center reaches `0.000718` in 100 generations — a 700× reduction, in about a second of wall clock on
the NumPy backend (interpreter start and tracing included).

Because `param_shapes` is the layout, slicing the center back into tensors is a `cumsum` away:

```python
weights = np.asarray(state.algorithm_state.center.numpy(), dtype=np.float32)
bounds = np.cumsum([0] + [int(np.prod(shape)) for shape in param_shapes])
network = [weights[bounds[i]:bounds[i + 1]].reshape(shape) for i, shape in enumerate(param_shapes)]
print([w.shape for w in network])      # [(8, 4), (8,), (1, 8), (1,)]
```

## What the problem evaluates

`evaluate` is a plain traced function with the **payload protocol**
`(center_flat, seeds, sigma)`:

| Payload slot | Shape / dtype | Meaning |
|---|---|---|
| `center_flat` | `(dim,)` float32 | The flattened network parameters — the algorithm's search point |
| `seeds` | `(pop_size,)` int64 | One deterministic seed per perturbed individual |
| `sigma` | static Python float | Perturbation scale; the algorithm passes its `noise_stdev` |

It returns `(fitness, problem_state)` where `fitness` is `(pop_size,)` float32 with minimization
semantics, and `problem_state` is the shared empty `ProblemState` (the problem is stateless).

Inside the graph, for every individual `i`:

1. **Regenerate its noise from its seed.** `evoxc` keys are stateless and there is no jump-ahead
   counter stream, so the noise is a deterministic **splitmix64 integer hash + Box–Muller** transform
   keyed by `(seed_i, element offset)`. The same code path lives in
   `evox_etl.algorithms.so.es_variants.virtual_noise`, which is what makes algorithm and problem
   agree on the perturbation without ever materializing it.
2. **Run the perturbed forward pass.** Each `Linear` layer computes the shared center matmul plus a
   `sigma`-scaled noise matmul — the `(out, in)` weight delta is never added to the center weight —
   followed by the layer's activation.
3. **Score it.** The baked `(batch, in_features)` input is pushed through all layers, and the
   per-sample `mse`/`cross_entropy` loss is reduced over the batch.

`VirtualES.step` then turns those `(pop_size,)` fitness values into a search direction by weighting
the regenerated noise with the fitness values
(`flat_grad = Σ_i fitness_i · noise_i / (pop_size · sigma)`), i.e. the standard Gaussian-smoothing
estimate of the gradient of the smoothed loss, and moves the center along it (optionally with Adam:
`make_virtual_es(..., optimizer="adam")`).

```{note}
Nothing in that pipeline is `(pop_size, dim)`-sized in memory: the state is `center (dim,)`,
`seeds (pop_size,)`, two Adam moments and the key. That is what makes the family usable on networks
with millions of parameters — the cost of a generation is one batched forward pass, not a
population of full models.
```

```{tip}
`workflow.monitor.plot(animation=False)` plots the per-generation `Min`/`Max`/`Median`/`Average` of
the **perturbed** fitness values, which is the noisier curve you should expect. The monitor's
solution history for this family is the broadcast center (every row identical), so read the center
from `state.algorithm_state.center`, not from `get_best_solution()`.
```

## The LoRA variant

`make_virtual_lora_es` replaces the full per-individual weight noise with a rank-`r` factorization:
the weight delta becomes `sigma · (h @ Aᵀ) @ Bᵀ` with `A (pop, r, in)` and `B (pop, out, r)`, so a
layer needs `r · (out + in)` noise numbers instead of `out · in`. The problem knows about it through
the same `lora_rank` field:

```python
from evox_etl.algorithms import make_virtual_lora_es

lora_problem = make_virtual_problem(
    param_shapes=param_shapes,
    layer_specs=layer_specs,
    inputs=inputs.reshape(-1).tolist(),
    targets=targets.reshape(-1).tolist(),
    in_features=in_features,
    out_features=out_features,
    batch_size=batch,
    loss="mse",
    reduction="mean",
    lora_rank=2,                                  # <- the only difference
)
lora_algorithm = make_virtual_lora_es(
    param_shapes,
    lora_rank=2,
    pop_size=256,
    center_init=(0.0,) * dim,
    learning_rate=0.1,
    noise_stdev=0.1,
)
lora_workflow = StdWorkflow(
    lora_algorithm, lora_problem,
    monitor=EvalMonitorConfig(),
    opt_direction="min",
    num_generations=100,
    backend="numpy",
)
lora_state = lora_workflow.run(seed=42)
print(f"LoRA best noisy fitness: {float(lora_state.algorithm_state.best_fitness.numpy()):.6f}")
print(f"LoRA center loss after : {center_loss(lora_problem, lora_state.algorithm_state.center.numpy()):.6f}")
```

```text
LoRA best noisy fitness: 0.000739
LoRA center loss after : 0.004241
```

The rank-2 run also learns (loss `0.519039 → 0.004241`) but lands about 6× worse than the full-noise
run on this tiny network, which is expected: a `(1, 8)` weight needs only 8 noise numbers, while
rank 2 needs `2 · (1 + 8) = 18`. LoRA pays off on **wide** layers, where `r · (out + in) ≪ out · in`
— that is why the family scales to large models, and the low-rank variant is also what an
`evox_etl_ext` extension would build on.

## Adapting it to your own data

1. **Replace the dataset.** `inputs`/`targets` are flat float tuples baked into the graph: a
   `batch_size × in_features` block of inputs and either `batch_size × out_features` targets (`mse`)
   or `batch_size` class indices (`cross_entropy`). Loading them with NumPy host-side is fine — only
   the traced code must stay torch/NumPy-free.
2. **Describe your architecture.** `param_shapes` + `layer_specs` for the `Linear` stack; the layer
   chain is validated against `in_features`/`out_features`, so a mismatch raises `ValueError` at
   construction time rather than at trace time.
3. **Pick the budget.** `pop_size`, `noise_stdev` and `learning_rate` are the three knobs that matter;
   `learning_rate=0.1` with `noise_stdev=0.1` is a safe starting point, and `optimizer="adam"` helps
   when the loss surface has very different scales per layer.

```{note}
The dataset is one fixed batch per generation by design — there is no shuffling inside the traced
problem. For mini-batch training, build several problem configs (one per batch) and rotate them
between runs, or feed a larger baked batch and rely on `reduction="mean"`.
```

## Where to go next

- The single-objective workflow these algorithms plug into: {doc}`so`.
- Multi-objective optimization and front metrics: {doc}`mo`.
- Writing your own `Problem` config (e.g. wrapping a simulator rollout): {doc}`../guide/developer/custom-alg-pro`.
- Backends — the same virtual-ES program on IREE/TVM/XLA devices: {doc}`../tutorial/tutorial_part6`.

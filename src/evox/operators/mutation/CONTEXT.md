# Mutation Operators (`src/evox/operators/mutation/`)

## Intent
Provides mutation operators for EvoX's evolutionary computation pipeline. Mutation introduces diversity into a population by perturbing individual decision variables. All operators are **pure functions** operating on PyTorch tensors — they hold no state, require no `ModuleBase` inheritance, and run on GPU.

## API Surface

### `polynomial_mutation(x, lb, ub, pro_m=1, dis_m=20) -> torch.Tensor`

Polynomial Mutation (PM), inspired by PlatEMO. The canonical real-valued mutation operator used across all multi-objective algorithms in EvoX (NSGA-II, NSGA-III, MOEA/D, RVEA, RVEA*, HypE).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | `torch.Tensor` | *(required)* | Population, shape `(n, d)` — `n` individuals, `d` decision variables |
| `lb` | `torch.Tensor` | *(required)* | Lower bounds for each decision variable |
| `ub` | `torch.Tensor` | *(required)* | Upper bounds for each decision variable |
| `pro_m` | `float` | `1` | Mutation probability. A site mutates when `rand() < pro_m / d`, giving ~1 expected mutation per individual by default |
| `dis_m` | `float` | `20` | Distribution index. Higher values produce offspring closer to the parent (narrower perturbation) |

**Returns**: Mutated population tensor of shape `(n, d)`, clipped to `[lb, ub]`.

**Usage pattern** — algorithms bind it directly as a callable:
```python
self.mutation = polynomial_mutation  # defaults used
# or wrapped with partial parameters:
self.mutation = lambda x: polynomial_mutation(x, lb=self.lb, ub=self.ub, pro_m=0.5)
```

### Exports
- `polynomial_mutation` (the only public symbol, declared in `__init__.py` via `__all__`)

## Constraints
- **Stateless & pure**: No internal state, no `nn.Module` or `ModuleBase` inheritance. Functions are side-effect-free.
- **JIT-compatible**: Uses `evox.utils.maximum` / `evox.utils.minimum` (not `torch.max` / `torch.min`) to ensure TorchScript traceability.
- **Same-device**: All input tensors (`x`, `lb`, `ub`) must reside on the same device. The output inherits `x.device`.
- **Bounds enforcement**: The operator internally clips to `[lb, ub]` via `maximum(minimum(x, ub), lb)` before computing perturbations.

## Files
| File | Purpose |
|------|---------|
| `pm_mutation.py` | Implementation of `polynomial_mutation` |
| `__init__.py` | Re-exports `polynomial_mutation` as the public API |

## Role in the Evolutionary Pipeline
Mutation sits alongside **crossover** and **selection** under `src/evox/operators/`. It is consumed exclusively by multi-objective algorithms (`src/evox/algorithms/mo/`), where it serves as the default (and currently sole) mutation operator. Algorithms treat mutation as a pluggable callable — `polynomial_mutation` is the default, but any function matching the same signature can be substituted.

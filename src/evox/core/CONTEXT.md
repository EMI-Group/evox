# Intent

The `core` module provides the foundational types and utilities for EvoX. It defines the base class hierarchy (`ModuleBase` → components) and essential wrappers that adapt PyTorch's `torch.nn.Module`, `torch.compile`, `torch.vmap`, and `torch.func.functional_call` for evolutionary computation workloads. Everything else in EvoX builds on these abstractions.

# API Surface

## Re-exported via `evox.core` (and `evox` top-level)

| Symbol | Source | Purpose |
|---|---|---|
| `ModuleBase` | `module.py` | Base module (extends `nn.Module`); all components inherit from this |
| `Parameter` | `module.py` | Wraps `nn.Parameter` with `requires_grad=False` by default; marks hyperparameters |
| `Mutable` | `module.py` | Wraps `nn.Buffer`; marks tensors that mutate during iteration |
| `compile` | `module.py` | Fixed `torch.compile` wrapper (paper over PyTorch issue #124423) |
| `vmap` | `module.py` | Fixed `torch.vmap` wrapper (same scalar-index workaround) |
| `use_state` | `module.py` | Functional transform: converts a module/method into a stateful function accepting `params_and_buffers` |
| `Agent` | `components.py` | Base class for individual RL-style agents |
| `Algorithm` | `components.py` | Base class for evolutionary algorithms |
| `Problem` | `components.py` | Base class for fitness evaluation problems |
| `Workflow` | `components.py` | Base class for algorithm–problem orchestration |
| `Monitor` | `components.py` | Base class for evolutionary process monitoring callbacks |

## Internal (not publicly exported)

| Symbol | File | Purpose |
|---|---|---|
| `TransformGetSetItemToIndex` | `module.py` | `TorchFunctionMode` that converts scalar tensor indices to 1D tensors for `__getitem__`/`__setitem__` |
| `_ReplaceForwardModule` | `module.py` | Internal helper: wraps a module with a replacement `forward` for `use_state` |

# Component Hierarchy

```
torch.nn.Module
  └── ModuleBase          (train(False) enforced; eval() disabled)
        ├── Algorithm     (step, init_step, final_step, evaluate, record_step)
        ├── Problem       (evaluate)
        ├── Workflow      (step, init_step, final_step)
        ├── Agent         (act)
        └── Monitor       (post_ask, pre_eval, post_eval, pre_tell, set_config, record_auxiliary)
```

All component base classes are abstract (`ABC`). They define the interface contracts that concrete implementations in `algorithms/`, `problems/`, `workflows/`, etc. must fulfill.

# Constraints & Design Notes

- **Always in "train" mode**: `ModuleBase.__init__` calls `self.train(False)` to set training mode off by default. `eval()` is deliberately disabled (`assert False`) because the train/eval distinction is ambiguous for evolutionary computation modules.
- **No gradient flow**: `Parameter` defaults to `requires_grad=False`. EvoX uses evolutionary strategies, not backpropagation — parameters are hyperparameters, not learnable weights.
- **Functional transformation**: `use_state` enables `torch.func.functional_call` semantics on arbitrary `nn.Module`s, which is essential for `vmap`-ing workflows over batched states. The signature is `(params_and_buffers, *args, **kwargs) -> params_and_buffers | (params_and_buffers, output)`.
- **PyTorch compat patches**: `compile` and `vmap` work around [PyTorch issue #124423](https://github.com/pytorch/pytorch/issues/124423) where scalar tensor indices are implicitly converted to Python scalars, causing graph breaks under `torch.compile` and errors under `torch.vmap`. The `TransformGetSetItemToIndex` mode intercepts `__getitem__`/`__setitem__` and converts scalar indices to 1D tensors.
- **Monitor lives outside jit**: Monitors are designed to run outside the compiled workflow graph. They receive callbacks at specific lifecycle points but are not part of the `torch.compile` trace.

# Routing Table

This is a leaf module with no child subdirectories. Its public API is consumed by:

- `src/evox/algorithms/` → Concrete algorithm implementations (extend `Algorithm`)
- `src/evox/problems/` → Concrete problem implementations (extend `Problem`)
- `src/evox/workflows/` → Concrete workflow implementations (extend `Workflow`)
- `src/evox/operators/` → Evolutionary operators using `Parameter`/`Mutable`
- `src/evox/utils/` → Utilities potentially using `compile`, `vmap`, `use_state`
- `src/evox/` (top-level) → Re-exports core symbols for user-facing API

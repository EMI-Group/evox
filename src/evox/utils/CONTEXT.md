## Intent
The `evox.utils` module provides **compiler-compatibility utilities** that bridge the gap between EvoX's evolutionary computation logic and PyTorch's graph compilation (`torch.compile`) and vectorized-mapping (`torch.vmap`) infrastructure.
Many standard PyTorch operators and Python control-flow constructs are not yet supported in operator fusion or vmap; this module supplies drop-in replacements and registration primitives to make EvoX fully compatible with these features.
All functions defined here have **limitations** (e.g., numerical precision tradeoffs, no dynamic shapes, no direct vmap of Python loops). See individual docstrings for details.

## API Surface
The module exports four categories of utilities (single source of truth for exports: `__init__.py` `__all__`).

### 1. Compiler/Vmap Fix Operators (`jit_fix_operator.py`)
Reimplementations of PyTorch operators that are not yet supported in `torch.compile`'s operator fusion or are not present in PyTorch.
Each uses only fusion-compatible primitives (e.g., `torch.relu` instead of `torch.where`/branching).

| Export | Purpose |
|---|---|
| `switch(label, values)` | Element-wise selection from a list of tensors by integer label (vmap-compatible alternative to if-else) |
| `clamp(a, lb, ub)` | Clamp tensor `a` between tensor bounds `lb` and `ub` |
| `clamp_float(a, lb, ub)` | Clamp tensor `a` between float scalar bounds |
| `clamp_int(a, lb, ub)` | Clamp tensor `a` between int scalar bounds |
| `clip(a)` | Single-arg `clamp(a, 0, 1)` — clamps into [0, 1]; it is NOT a `clamp` alias |
| `maximum(a, b)` / `minimum(a, b)` | Element-wise max/min of two tensors |
| `maximum_float(a, b)` / `minimum_float(a, b)` | Element-wise max/min of tensor and float scalar |
| `maximum_int(a, b)` / `minimum_int(a, b)` | Element-wise max/min of tensor and int scalar |
| `lexsort(keys, dim=-1)` | Lexicographical sort across multiple key tensors (NumPy-compatible) |
| `nanmin(input_tensor, dim=-1, keepdim=False)` | Minimum ignoring NaN values |
| `nanmax(input_tensor, dim=-1, keepdim=False)` | Maximum ignoring NaN values |
| `randint(low, high, size, dtype=None, device=None, generator=None)` | Random int tensor with tensor (dynamic) bounds |

**Key constraint**: These use `torch.relu`-based arithmetic instead of boolean branching (e.g., `clamp` is `a + relu(lb-a) - relu(a-ub)`), which is fusion-friendly but may lose precision for float tensors.
Use the native PyTorch ops when precise results are required and fusion is not needed.

### 2. Custom Operator Registration (`op_register.py`)
| Export | Purpose |
|---|---|
| `register_vmap_op(fn, *, fake_fn, vmap_fn, fake_vmap_fn, vmap_wrap_inputs, vmap_out_dims=0, max_vmap_level=None, name=None, mutates_args=(), device_types=None, schema=None)` | Decorator/function to register a custom operator with optional vmap support via `torch.library.custom_op` |
| `VmapInfo` | Re-export of `torch._functorch.autograd_function.VmapInfo` (see below) |

`register_vmap_op` wraps `torch.library.custom_op` and layers on vmap registration; it provides `_default_vmap_wrap_inputs`, which moves vmap dims to position 0 and handles pytree structures.
Required callbacks:
- `fake_fn` — abstract evaluation for the operator (required; `assert fake_fn is not None`).
- `vmap_fn` — vmap implementation (optional; if provided, enables vmap support).
- `fake_vmap_fn` — abstract evaluation for the vmap variant (required if `vmap_fn` is given).
- `vmap_wrap_inputs` — custom input-preprocessing for vmap (defaults to `_default_vmap_wrap_inputs`).
- `max_vmap_level` — vmap nesting depth (defaults to 1 when `vmap_fn` is given); it registers `max_vmap_level` extra custom ops named `<name>_vmap`, `<name>_vmap_vmap`, ...
- Default `name` is `"evox::_custom_op_" + fn.__name__`.

`VmapInfo` is exactly `VmapInfo(batch_size: int, randomness: str)`; it is passed as the first positional argument to every `vmap_fn`/`fake_vmap_fn`/`vmap_wrap_inputs` callback.
`problems/neuroevolution/brax.py` and `mujoco_playground.py` import it from `evox.utils` and read `vmap_info.batch_size`.

### 3. Parameter/Vector Conversion (`parameters_and_vector.py`)
| Export | Purpose |
|---|---|
| `ParamsAndVector(dummy_model)` | `ModuleBase` that converts between PyTorch parameter dicts and flat vectors |

It flattens a model's named parameters into one 1D vector (for mutation/crossover) and reconstructs the parameter dict from a vector.
- `to_vector(params)` / `to_params(vector)` — single model.
- `batched_to_vector(batched_params)` / `batched_to_params(vectors)` — batched (population) models, batch on dim 0.
- `forward(x)` — alias for `batched_to_params`, compatible with `StdWorkflow`.

#### `ParamsAndVector` exact semantics
- `__init__(self, dummy_model: nn.Module)` sets exactly four attributes: `self.param_spec` (a `torch.utils._pytree.TreeSpec` whose context is `list(dict(dummy_model.named_parameters()).keys())`), `self.shapes` (tuple of `torch.Size`), `self.start_indices` (tuple of int offsets), `self.slice_sizes` (tuple of int, `math.prod(shape)` via `from math import prod`).
- Ordering is **torch pytree dict insertion order**, i.e. `dict(model.named_parameters())` order, i.e. `named_parameters()` order (module's own params first, then children recursively) with dotted names; nothing is sorted (`_dict_flatten` is `return list(d.values()), list(d.keys())`).
- Flatten: `to_vector` = `torch.concat([x.reshape(-1) for x in leaves], dim=0)`; `batched_to_vector` = `torch.concat([x.reshape((x.size(0), -1)) for x in leaves], dim=1)`.
- Unflatten loop: `vector.narrow(dim=0, start=start_index, length=slice_size).reshape(shape)`; batched variant uses `narrow(dim=1, ...)` then `reshape(batch_size, *shape)`.
- Only `named_parameters()` are included — **buffers are excluded** (contrast `torch.func.stack_module_state`, which returns `params | buffers` and is what `problems/neuroevolution/utils.py` uses).
- No clone/detach/`contiguous()`/dtype/device casting anywhere: `to_params` returns **views aliasing the input vector's storage** (in-place writes propagate back), on `vector.device`/`vector.dtype`, as plain tensors (not `nn.Parameter`).
- `to_vector` on real `nn.Parameter` leaves (`requires_grad=True`) returns a graph-connected tensor — callers must `.detach()`.
- No key validation: a dict with the same keys in a different order is silently concatenated in *that* dict's order and unflattened in `param_spec` order, giving silently wrong tensors whenever shapes differ.
- As a `ModuleBase`, `__init__` runs `nn.Module.__init__` + `self.train(False)`; `param_spec`/`shapes` stay plain Python attributes (not buffers).

### 4. PyTree Re-Exports (`re_export.py`)
| Export | Purpose |
|---|---|
| `tree_flatten(tree)` | Flatten an arbitrary pytree; returns `(leaves: list, TreeSpec)` (2-tuple, unlike JAX's 3-tuple) |
| `tree_unflatten(leaves, spec)` | Reconstruct a pytree from flat leaves and a tree spec |

These are thin re-exports from `torch.utils._pytree` that also patch `nn.Buffer` into `torch.nn` if absent (compatibility shim).

### Absent helpers (do not assume they exist)
- `FlattenedState` does **not** exist anywhere in this repository (no class, no function).
- There is **no activation-lookup helper** here (no `get_activation`/`ACTIVATION_MAP`); `problems/neuroevolution/virtual_problem.py` and `virtual_lora_problem.py` dispatch activations by `isinstance` checks on `nn.ReLU`/`nn.Tanh`/... modules directly.
- `compile`/`vmap`/`use_state`/`Parameter`/`Mutable`/`ModuleBase` live in `evox.core`, not here; the closest thing to a "flattened state" is `ParamsAndVector` itself plus `torch.func.stack_module_state` in `problems/neuroevolution/utils.py`.

## Constraints
- No direct control flow (`if`/`else`, `for`, `while`) should be used in code that will be vmapped — use `torch.cond` and `torch.while_loop` instead.
- The fix operators use relu-based arithmetic and are NOT exact replacements for their PyTorch counterparts when floating-point tensors are involved.
- `randint` with dynamic bounds MUST be used with `torch.compile(..., dynamic=False)` when compiled.
- `register_vmap_op` requires `fake_fn`, and `fake_vmap_fn` whenever `vmap_fn` is given — these cannot be `None`.
- When registering vmap ops, mutation arguments (`mutates_args`) MUST be accurately declared or behavior is undefined.

## Routing Table
| Area | Path | Description |
|---|---|---|
| Fix operators | `jit_fix_operator.py` | relu-based `torch.compile`/vmap-friendly replacements (`switch`, `clamp*`, `maximum*`, `lexsort`, `nanmin`/`nanmax`, `randint`) |
| Op registration | `op_register.py` | `register_vmap_op` + `VmapInfo` re-export for `torch.library.custom_op` with nested vmap support |
| Param↔vector | `parameters_and_vector.py` | `ParamsAndVector` adapter used by neuroevolution problems and `StdWorkflow.forward` |
| PyTree shim | `re_export.py` | `tree_flatten`/`tree_unflatten` re-exports + `nn.Buffer` backport |

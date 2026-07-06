## Intent

The `evox.utils` module provides **compiler-compatibility utilities** that bridge the gap between EvoX's evolutionary computation logic and PyTorch's graph compilation (`torch.compile`) and vectorized-mapping (`torch.vmap`) infrastructure. Many standard PyTorch operators and Python control-flow constructs are not yet supported in operator fusion or vmap; this module supplies drop-in replacements and registration primitives to make EvoX fully compatible with these features.

All functions defined here have **limitations** (e.g., numerical precision tradeoffs, no dynamic shapes, no direct vmap of Python loops). See individual docstrings for details.

## API Surface

The module exports four categories of utilities:

### 1. Compiler/Vmap Fix Operators (`jit_fix_operator.py`)

Reimplementations of PyTorch operators that are not yet supported in `torch.compile`'s operator fusion or are not present in PyTorch. Each uses only fusion-compatible primitives (e.g., `torch.relu` instead of `torch.where`/branching):

| Export | Purpose |
|---|---|
| `switch(label, values)` | Element-wise selection from a list of tensors by integer label (vmap-compatible alternative to if-else) |
| `clamp(a, lb, ub)` | Clamp tensor `a` between tensor bounds `lb` and `ub` |
| `clamp_float(a, lb, ub)` | Clamp tensor `a` between float scalar bounds |
| `clamp_int(a, lb, ub)` | Clamp tensor `a` between int scalar bounds |
| `clip(a, lb, ub)` | Alias for `clamp` |
| `maximum(a, b)` / `minimum(a, b)` | Element-wise max/min of two tensors |
| `maximum_float(a, b)` / `minimum_float(a, b)` | Element-wise max/min of tensor and float scalar |
| `maximum_int(a, b)` / `minimum_int(a, b)` | Element-wise max/min of tensor and int scalar |
| `lexsort(keys, dim)` | Lexicographical sort across multiple key tensors (NumPy-compatible) |
| `nanmin(input, dim, keepdim)` | Minimum ignoring NaN values |
| `nanmax(input, dim, keepdim)` | Maximum ignoring NaN values |
| `randint(low, high, size)` | Random int tensor with tensor-valued bounds (dynamic range) |

**Key constraint**: These use `torch.relu`-based arithmetic instead of boolean branching (e.g., `clamp` is `a + relu(lb-a) - relu(a-ub)`), which is fusion-friendly but may lose precision for float tensors. Use the native PyTorch ops when precise results are required and fusion is not needed.

### 2. Custom Operator Registration (`op_register.py`)

| Export | Purpose |
|---|---|
| `register_vmap_op(fn, *, fake_fn, vmap_fn, fake_vmap_fn, ...)` | Decorator/function to register a custom operator with optional vmap support via `torch.library.custom_op` |
| `VmapInfo` | Named tuple re-exported from `torch._functorch.autograd_function.VmapInfo` |

`register_vmap_op` wraps `torch.library.custom_op` and layers on vmap registration. It provides a `_default_vmap_wrap_inputs` that moves vmap dimensions to position 0 and handles pytree structures. Required callbacks:
- `fake_fn` — abstract evaluation for the operator (required)
- `vmap_fn` — vmap implementation (optional; if provided, enables vmap support)
- `fake_vmap_fn` — abstract evaluation for the vmap variant (required if `vmap_fn` is given)
- `vmap_wrap_inputs` — custom input-preprocessing for vmap (defaults to `_default_vmap_wrap_inputs`)
- `max_vmap_level` — vmap nesting depth (defaults to 1)

### 3. Parameter/Vector Conversion (`parameters_and_vector.py`)

| Export | Purpose |
|---|---|
| `ParamsAndVector(dummy_model)` | ModuleBase class that converts between PyTorch model parameter dicts and flat vectors |

This is a core utility for evolutionary algorithms: it flattens a model's named parameters into a single 1D vector (for mutation/crossover) and reconstructs the parameter dict from a vector. Supports both single and batched modes:
- `to_vector(params)` / `to_params(vector)` — single model
- `batched_to_vector(batched_params)` / `batched_to_params(vectors)` — batched (population) models
- `forward(x)` — alias for `batched_to_params`, compatible with `StdWorkflow`

Internally uses `tree_flatten`/`tree_unflatten` to handle arbitrary `nn.Module` parameter trees.

### 4. PyTree Re-Exports (`re_export.py`)

| Export | Purpose |
|---|---|
| `tree_flatten(tree)` | Flatten an arbitrary pytree structure |
| `tree_unflatten(leaves, spec)` | Reconstruct a pytree from flat leaves and a tree spec |

These are thin re-exports from `torch.utils._pytree` that also patch `nn.Buffer` into `torch.nn` if absent (compatibility shim).

## Constraints

- No direct control flow (`if`/`else`, `for`, `while`) should be used in code that will be vmapped — use `torch.cond` and `torch.while_loop` instead.
- The fix operators use relu-based arithmetic and are NOT exact replacements for their PyTorch counterparts when floating-point tensors are involved.
- `randint` with dynamic bounds MUST be used with `torch.compile(..., dynamic=False)` when compiled.
- `register_vmap_op` requires both `fake_fn` and (if vmap is enabled) `fake_vmap_fn` — these cannot be `None`.
- When registering vmap ops, mutation arguments (`mutates_args`) MUST be accurately declared or behavior is undefined.

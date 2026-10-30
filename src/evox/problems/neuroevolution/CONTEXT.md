# Neuroevolution Problems

## Intent
Provides reinforcement learning / policy-search evaluation domains for EvoX. Every problem wraps a neural network policy and evaluates it against an environment or dataset, returning scalar fitness (reward or loss). These problems enable evolutionary optimization of neural network weights for robotics, control, and supervised learning tasks.

## API Surface
This directory is a flat module (no child subdirectories) exposing three Problem classes and one shared utility:

- **`brax.py`** → `BraxProblem(Problem)` — Evaluates policies in Google Brax physics simulation environments (e.g., Swimmer, Ant, HalfCheetah). Runs each individual for `num_episodes` episodes per evaluation, aggregates rewards via `reduce_fn` (default `torch.mean`). Supports visualization via `visualize()` (HTML or rgb_array).
- **`mujoco_playground.py`** → `MujocoProblem(Problem)` — Evaluates policies in MuJoCo Playground (MJX) physics environments (e.g., SwimmerSwimmer6, Go1JoystickFlatTerrain). Near-identical architecture to `BraxProblem`. Supports visualization via `visualize()` (mp4 or gif).
- **`supervised_learning.py`** → `SupervisedLearningProblem(Problem)` — Evaluates model parameters against a `DataLoader` + loss `criterion`. Supports single-sample and vmapped (batched) evaluation. `n_batch_per_eval` controls batch count; `reduction` ("mean" | "sum") aggregates per-batch losses.
- **`virtual_problem.py`** → `VirtualProblem(Problem)` (with backward-compatible alias `VirtualLoRAProblem`) — Evaluates a virtual Gaussian-noise-perturbed population WITHOUT materializing the full perturbed weights. `evaluate()` receives a tuple `(center_flat, seeds, sigma)` instead of a full population; full Gaussian-noise perturbations of both weight and bias are applied layer-by-layer during the forward pass via the fused kernel `virtual_perturbed_linear` (base matmul + on-the-fly per-individual noise from a seed). Supports `nn.Sequential` models with `nn.Linear` layers and common activations (ReLU, Tanh, Sigmoid, GELU, LeakyReLU, ELU, Softmax, Identity). Does NOT support HPO wrapper.
- **`virtual_lora_problem.py`** → Legacy `VirtualLoRAProblem(Problem)` — The original LoRA-based implementation. Superseded by `virtual_problem.py`; kept for reference (no longer exported by `__init__.py`).
  The alias `VirtualLoRAProblem = VirtualProblem` at the end of `virtual_problem.py` shadows it: `evox.problems.neuroevolution.VirtualLoRAProblem` resolves to the full-noise class, so the LoRA class must be imported as `evox.problems.neuroevolution.virtual_lora_problem.VirtualLoRAProblem`.
- **`utils.py`** → Shared utility: `ModelStateForwardResult` (NamedTuple: `init_state`, `state_forward`) and `get_vmap_model_state_forward(model, pop_size, device)` which uses `torch.func.stack_module_state` + `evox.core.vmap` to create vmapped state + forward callables for batched policy evaluation. Used by `brax.py`, `mujoco_playground.py`, and `supervised_learning.py`.

## Architecture: Torch↔JAX Bridge (Brax & MuJoCo)

`BraxProblem` and `MujocoProblem` use a **Torch↔JAX DLPack bridge** for GPU-accelerated physics simulation:

- **`to_jax_array(x: torch.Tensor) -> jax.Array`**: Converts via `jax.dlpack.from_dlpack`. Handles `is_batchedtensor` unwrapping and CPU-fallback when JAX has no GPU support.
- **`from_jax_array(x: jax.Array, device) -> torch.Tensor`**: Converts via `torch.utils.dlpack.from_dlpack`.
- Both conversion functions are duplicated in `brax.py` and `mujoco_playground.py` (no shared implementation).

### Evaluation Loop (per module)
Each module defines a module-level global dict (`__brax_data__` / `__mjx_data__`) keyed by Python `id(self)`, storing pre-compiled vmap'd JAX functions (reset, step, state_forward) and state keys. A `weakref.finalize` ensures cleanup when the Problem instance is garbage-collected.

The core evaluation loop (`_evaluate_brax_main` / `_evaluate_mjx_main`) is registered as a **custom Torch operator** (`torch.library.custom_op`) with:
- A **fake implementation** (for `torch.compile` tracing — returns empty tensors with correct shapes)
- A **vmap registration** (for `torch.func.vmap` — flattens vmap+pop dims, calls main loop, unflattens)

This allows the JAX physics step to be treated as an opaque Torch operator that `torch.compile` and `torch.func.vmap` can reason about.

### State Management
- **Vmapped path** (`evaluate`): Uses `self.vmap_init_state` (stacked params from `torch.func.stack_module_state`) + `self.vmap_state_forward` (vmapped `use_state(model)`). Policy parameters are merged into init state via dict union.
- **Single path** (`visualize` / `_evaluate_*_record`): Uses `self.init_state` (single `state_dict()`) + `self.state_forward` (unvmapped `use_state(model)`).

## Shared Interface (Problem Contract)
All three classes implement `evaluate(self, pop_params: Dict[str, nn.Parameter]) -> torch.Tensor`:
- `pop_params`: Dictionary mapping parameter names to tensors of shape `(batch_size, *param_shape)`.
- Returns: Tensor of shape `(batch_size,)` with per-individual fitness (reward or loss).

## Virtual-Problem Semantics (parity-critical)
`VirtualProblem.evaluate(self, payload)` / `VirtualLoRAProblem.evaluate(self, payload)` take one positional `payload = (center_flat, seeds, sigma)`, not a population dict or tensor.
`center_flat` is `(dim,)` float32; `seeds` is `(pop_size,)` int64; `sigma` is a Python float; `pop_size` is inferred as `seeds.shape[0]`.
Both return a plain `(pop_size,)` fitness tensor and **no** problem state; both are minimizers.
Flat-vector ordering is torch `named_parameters()` order via `ParamsAndVector` from `evox.utils`: weight blocks are `(out_features, in_features)` row-major, each `nn.Linear` bias block immediately follows its weight block, so `offsets[bias] == offsets[weight] + out*in`.
`n_batch_per_eval == -1` iterates the whole `data_loader`; any other value draws that many batches from the persistent `self.data_loader_iter` created in `__init__`, restarting it on `StopIteration` — consecutive evaluations therefore see successive batches (stateful round-robin).
`reduction` is only `"mean"` (default) or `"sum"`; any other value silently takes the `"sum"` branch.
A scalar criterion output triggers `warnings.warn` and is expanded to `(pop_size,)`; non-scalar per-sample losses are reshaped to `(pop_size, batch)` before reduction.
Activations are a module-type whitelist `_SUPPORTED_ACTIVATIONS` (ReLU, Tanh, Sigmoid, GELU, LeakyReLU, ELU, Softmax, Identity) checked with `isinstance`; any other child raises `NotImplementedError`.
Noise is not `torch.randn`: `VirtualProblem` uses splitmix64 + Box-Muller `_cpu_normal_noise` keyed by `(seed_i, element_index = offset + flat_index)`; `VirtualLoRAProblem` uses `philox_normal` counter offsets (bit-identical across Triton/PyTorch, whereas `virtual_perturbed_linear`'s Triton path uses `tl.randn` and is NOT numerically equal to its fallback).
Full-noise path: `virtual_perturbed_linear(h, weight, bias, seeds, sigma, offset)` perturbs weight and bias together and computes `bias_offset = offset + out*in` internally, so only the weight block's offset is passed.
LoRA path: weight delta is `sigma * (h @ Aᵀ) @ Bᵀ` with `A: (pop, rank, in)`, `B: (pop, out, rank)`; the bias gets direct noise `sigma * bias_noise` applied in the problem (no low-rank factor) from its own `counter_offsets[bias_idx]`.
First-layer arithmetic differs between the two: `virtual_problem` feeds 2-D `(batch, in)` into the fused op (`x_is_per_individual=False`), while `virtual_lora_problem` pre-expands to `(pop, batch, in)` and computes the base `h @ weight.t() + bias` itself.
Attributes set by both `__init__`s: `device`, `model`, `data_loader`, `data_loader_iter`, `criterion`, `n_batch_per_eval`, `reduction`, `param_names`, `param_shapes`, `params_and_vector`, `linear_param_indices`, plus `offsets` (virtual) or `counter_offsets` and `lora_rank` (LoRA).

## Constraints
- All problems inherit from `evox.core.Problem` and implement `evaluate(pop_params) -> Tensor`, **except** the two virtual problems, whose `evaluate(payload)` takes the `(center_flat, seeds, sigma)` tuple.
- **HPO Limitation**: None of the three problem classes support `HPOProblemWrapper` out-of-box. The JAX bridge prevents `torch.func.vmap` from working transparently.
  - *Workaround* (Brax/MuJoCo): Set `pop_size` to `inner_pop × outer_pop` and `num_repeats=1` in the HPO wrapper. Use `num_episodes` for statistical robustness instead.
- `BraxProblem` and `MujocoProblem` require both `torch` and `jax` installed with compatible DLPack support.
- Policy models must accept batched observations: `forward(batched_obs) -> action`.
- `BraxProblem` provides optional `torch.compile` on the policy via the `compile_policy` flag. `MujocoProblem` always compiles the single state_forward.
- `MujocoProblem.__init__` has no `backend` or `compile_policy` parameters (it takes `policy, env_name, max_episode_length, num_episodes, seed=None, pop_size=None, rotate_key=True, reduce_fn=torch.mean, device=None`), unlike `BraxProblem`.
- `get_vmap_model_state_forward` returns only the vmapped pair despite its docstring mentioning a `get_non_vmap` flag; the vmapped `init_state` is `params | buffers` from `torch.func.stack_module_state` (buffers included), unlike `ParamsAndVector` vectors (parameters only).
- `BraxProblem`/`MujocoProblem` keep a module-level registry `__brax_data__` / `__mjx_data__` keyed by `id(self)` and cleaned by `weakref.finalize`, so compiled JAX closures live outside the instance; `evaluate` also mutates `self.key` (PRNG state carried across calls) when `rotate_key=True`.

## Routing Table
- `./brax.py` → `BraxProblem` — Google Brax physics simulation evaluation
- `./mujoco_playground.py` → `MujocoProblem` — MuJoCo Playground (MJX) physics simulation evaluation
- `./supervised_learning.py` → `SupervisedLearningProblem` — Supervised learning loss-landscape evaluation
- `./virtual_problem.py` → `VirtualProblem` / `VirtualLoRAProblem` — Virtual Gaussian-noise population evaluation via fused `virtual_perturbed_linear` kernel (no full perturbation materialization)
- `./virtual_lora_problem.py` → Legacy `VirtualLoRAProblem` — Original LoRA-based implementation (not exported)
- `./utils.py` → `ModelStateForwardResult`, `get_vmap_model_state_forward` — Shared vmapped state/forward factory
- `./__init__.py` → Exports submodule names: `brax`, `mujoco_playground`, `supervised_learning`, `VirtualProblem`, and `VirtualLoRAProblem`

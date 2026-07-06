# Sampling Operators

## Intent
Provides stateless sampling/initialization operators for generating populations and reference vectors in EvoX. All operators are pure functions operating on PyTorch tensors, returning points in the unit hypercube or on the unit simplex.

## API Surface

| Function | Signature | Returns | Purpose |
|---|---|---|---|
| `uniform_sampling` | `(n: int, m: int) -> Tuple[Tensor, int]` | Points on unit simplex `(n_samples, m)`, count | Das & Dennis uniform simplex lattice design. Use: reference vectors for decomposition-based MOEAs (NSGA-III, MOEA/D). |
| `grid_sampling` | `(n: int, m: int) -> Tuple[Tensor, int]` | Grid points `(n_samples, m)`, count | Regular grid in unit hypercube `[0,1]^m`. Use: systematic population initialization. |
| `latin_hypercube_sampling_standard` | `(n: int, d: int, device: torch.device, smooth: bool = True) -> Tensor` | LHS points `(n, d)` | Standard Latin Hypercube Sampling in `[0,1]^d`. `smooth=True` places points at random positions within cells; `smooth=False` uses cell centers. |
| `latin_hypercube_sampling` | `(n: int, lb: Tensor, ub: Tensor, smooth: bool = True) -> Tensor` | LHS points `(n, d)` | LHS scaled to arbitrary hypercube `[lb, ub]`. Wraps `latin_hypercube_sampling_standard` with affine scaling. |

## Constraints
- All functions are **pure** (stateless) — no internal state, no randomness retained beyond the call.
- `uniform_sampling` uses combinatorial enumeration via `itertools.combinations` and can be expensive for large `n`/`m`.
- `grid_sampling` note: `n` is adjusted to `ceil(n^(1/m))` points per axis, so the actual number of samples may differ from `n`.
- `latin_hypercube_sampling` asserts that `lb` and `ub` share device, dtype, and shape (both 1D).

## Routing Table
- `uniform.py` → Uniform simplex sampling (Das & Dennis)
- `gird.py` → Regular grid sampling (note: filename has a typo — "gird" not "grid")
- `latin_hypercube.py` → Latin Hypercube Sampling (standard unit and bounded variants)

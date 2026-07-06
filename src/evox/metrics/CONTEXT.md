## Intent

The `evox.metrics` module provides GPU-accelerated evaluation metrics for multi-objective optimization. Each metric quantifies how well an approximated solution set matches the true Pareto front. All functions operate on `torch.Tensor` and are compatible with EvoX's PyTorch-based evolutionary computation pipeline.

## API Surface

| Function | File | Purpose |
|----------|------|---------|
| `gd(objs, pf)` | `gd.py` | **Generational Distance** — mean Euclidean distance from each solution to the nearest Pareto-front point. Lower is better. |
| `hv(objs, ref, num_sample)` | `hv.py` | **Hypervolume** — Monte Carlo estimate of the volume dominated by the solution set relative to a reference point. Higher is better. |
| `igd(objs, pf, p)` | `igd.py` | **Inverted Generational Distance** — mean distance from each Pareto-front point to the nearest solution. Measures both convergence and diversity. Lower is better. |

### Common Signatures

- `objs`: `(n, m)` tensor — `n` candidate solutions with `m` objectives.
- `pf`: `(k, m)` tensor — `k` reference points on the true Pareto front.
- `ref`: `(m,)` tensor — reference point for hypervolume (should be dominated by all solutions).
- All return scalar `torch.Tensor`.

### Extension Point

The module is designed to be extended by `evox_ext.metrics`. During autoload (`evox_ext.autoload_ext`), any additional metrics in `evox_ext.metrics` are merged into this namespace. Currently, the core module provides only the three foundational metrics above.

## Constraints

- Pure functions: no state, no side effects.
- PyTorch-only: no NumPy dependency; all computations use `torch` ops for GPU acceleration.
- No subdirectories; if the set of metrics grows significantly, consider grouping related metrics into submodules (e.g., convergence metrics vs. diversity metrics).

# vis_tools — Visualization & Serialization for EvoX

## Intent
Provides built-in visualization utilities and a binary serialization format for evolutionary optimization runs in EvoX. All plotting uses **Plotly** internally, producing interactive, animated figures for exploring population dynamics in both decision space and objective space. The serialization format (exv) enables efficient, streamable persistence of evolutionary runs.

## API Surface

### `plot.py` — Interactive Population Visualization
All functions return `plotly.graph_objects.Figure` and accept `**kwargs` forwarded to `plotly.Layout`.

| Function | Description |
|---|---|
| `plot_dec_space(population_history, **kwargs)` | Animated 2D scatter of population in **decision space** over generations. |
| `plot_obj_space_1d(fitness_history, animation=True, **kwargs)` | Single-objective fitness trends (min/max/median/avg). Dispatches to `_animation` or `_no_animation`. |
| `plot_obj_space_1d_no_animation(fitness_history, **kwargs)` | Static line chart of min, max, median, and average fitness across generations. |
| `plot_obj_space_1d_animation(fitness_history, **kwargs)` | Animated version of the above, building lines frame-by-frame. |
| `plot_obj_space_2d(fitness_history, problem_pf=None, sort_points=False, **kwargs)` | 2-objective fitness scatter, animated. Optionally overlays a **Pareto front** (`problem_pf`). |
| `plot_obj_space_3d(fitness_history, problem_pf=None, sort_points=False, **kwargs)` | 3-objective fitness 3D scatter, animated. Optionally overlays a **Pareto front**. |

All animated functions share a common pattern: Plotly `Frames` + `sliders` + Play/Pause `updatemenus`. Auto-ranging pads bounds by 5–10%.

### `exv.py` — EvoXVision Binary Format (exv)
A custom binary file format (`0x65787631` magic) for efficient streaming and random access of evolutionary run data.

| Symbol | Description |
|---|---|
| `_get_data_type(dtype)` | Maps `numpy.dtype` → exv type string (`"f32"`, `"f64"`, etc.). Internal helper. |
| `new_exv_metadata(population1, population2, fitness1, fitness2)` | Infers schema from the first two iterations and returns the metadata dict (population size, chunk size, per-field type/shape/offset). |
| `EvoXVisionAdapter(file_path, buffering=0)` | Stream writer for `.exv` files. Requires 2 iterations of data before writing: call `set_metadata()`, `write_header()`, then `write(*fields)` per iteration. `flush()` forces buffered writes. |

**File layout**: Magic (4B) → Header length (u32 LE) → JSON metadata (schema) → binary data chunks (population + fitness per iteration). Different schemas for initial vs. rest iterations.

## Constraints
- **Plotly required**: Plotting functions need `plotly` installed (not a hard EvoX dependency).
- **2D requirement**: `plot_dec_space` assumes 2-dimensional decision space (uses `pop[:, 0]` and `pop[:, 1]`).
- **Numpy types**: exv only supports fixed-width numeric dtypes (u8–u64, i16–i64, f16–f64).
- **exv requires 2 iterations**: Schema inference needs the first two iterations to detect differing initial/rest structures.

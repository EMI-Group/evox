# Visualization and result serialization

`evox.vis_tools` is a **host-side** toolkit: nothing in it is traced by `evoxc`. It consumes
the NumPy history lists a monitor recorded *after* a run — the per-generation fitness
(`EvalMonitor.fitness_history`) and, when enabled, the per-generation solutions
(`solution_history`) — and turns them into [Plotly](https://plotly.com/python/) figures or
EvoXVision (`.exv`) files. Figures are therefore always built **after** the compiled step graph
has finished; you never plot from inside it.

Plotly is an **optional** dependency (`pip install "evox[vis]"`). `import evox.vis_tools`
succeeds without it: `plot.py` guards its import (`go = None`) and only the `plot_*` functions
call `_require_plotly()`, which raises `ImportError: plotly is required for visualization but is
not installed. Install the optional dependency with: pip install "evox[vis]"`. `EvalMonitor.plot()`
degrades more gently — it *warns* and returns `None`. `evox.vis_tools.exv` never needs Plotly.

## The figure builders

All six builders live in `evox.vis_tools.plot`, are re-exported from `evox.vis_tools`,
and take a **list of per-generation NumPy arrays**:

| Function | Signature |
|---|---|
| `plot_dec_space` | `plot_dec_space(population_history, **kwargs)` |
| `plot_obj_space_1d` | `plot_obj_space_1d(fitness_history, animation=True, **kwargs)` |
| `plot_obj_space_1d_no_animation` | `plot_obj_space_1d_no_animation(fitness_history, **kwargs)` |
| `plot_obj_space_1d_animation` | `plot_obj_space_1d_animation(fitness_history, **kwargs)` |
| `plot_obj_space_2d` | `plot_obj_space_2d(fitness_history, problem_pf=None, sort_points=False, **kwargs)` |
| `plot_obj_space_3d` | `plot_obj_space_3d(fitness_history, problem_pf=None, sort_points=False, **kwargs)` |

- `plot_dec_space` animates the population in **decision space**, i.e. it reads the first two
  columns (`pop[:, 0]` / `pop[:, 1]`) of each generation, so the solutions must have `dim >= 2`.
  It adds one slider step per generation.
- `plot_obj_space_1d` is a dispatcher: `animation=True` (the default) forwards to
  `plot_obj_space_1d_animation`, `animation=False` to `plot_obj_space_1d_no_animation`.
  Both draw the per-generation **Min / Max / Median / Average** traces; the animation variant
  adds frames plus a generation slider.
- `plot_obj_space_2d` / `plot_obj_space_3d` are the multi-objective views (2 and 3 columns of
  each fitness row). `problem_pf=` overlays the true Pareto front as an extra `"Pareto Front"`
  scatter repeated in every frame; `sort_points=True` lexsorts each generation (`np.lexsort`)
  purely to make the animation look smoother. `**kwargs` go to the Plotly `go.Layout`
  constructor, so options such as `title="..."` work on every builder.

```python
from evox.algorithms import make_pso
from evox.problems.numerical import Sphere
from evox.vis_tools import plot_dec_space, plot_obj_space_1d_no_animation
from evox.workflows import EvalMonitorConfig, StdWorkflow

workflow = StdWorkflow(
    make_pso(pop_size=40, lb=[-5.0] * 4, ub=[5.0] * 4), Sphere(),
    monitor=EvalMonitorConfig(full_sol_history=True),  # keep the populations too
    opt_direction="min", num_generations=20, backend="numpy",
)
workflow.run()

pops = workflow.monitor.get_solution_history()   # list of (40, 4) arrays
fits = workflow.monitor.get_fitness_history()    # list of (40,) arrays

print(len(plot_dec_space(pops).layout.sliders[0].steps))   # 20 -> one step per generation
print(len(plot_obj_space_1d_no_animation(fits).frames))    # 0 -> no animation frames
```

## Plotting from the monitor: `EvalMonitor.plot`

`EvalMonitor.plot(problem_pf=None, source="eval", **kwargs)` is the shortcut used in practice.
It picks a builder from the number of objectives and forwards `**kwargs` to it:

| Objectives of `fitness_history[0]` | Builder used |
|---|---|
| 1 (1-D fitness array) | `plot_obj_space_1d(fitness_history, **kwargs)` |
| 2 | `plot_obj_space_2d(fitness_history, problem_pf, **kwargs)` |
| 3 | `plot_obj_space_3d(fitness_history, problem_pf, **kwargs)` |
| more | warns `Not supported yet.` and returns `None` |

`source=` selects which recorded channel is plotted:

- `source="eval"` (default) uses `get_fitness_history()` — the problem-evaluation side with the
  `opt_direction` scaling **undone**, so a maximization run shows the real (positive) values.
- `source="pop"` uses the raw `aux_history["fit"]` channel that the algorithm reported, i.e.
  **not** un-negated. It exists only for algorithms whose module has a recording hook and only
  when the monitor was created with `full_pop_history=True`.

It returns `None` — never a broken figure — in three cases: nothing was recorded
(`No fitness history recorded, return None`), Plotly is missing
(`No visualization tool available, return None. Hint: pip install "evox[vis]"`), or the
objective count is unsupported. An unknown `source` raises `ValueError` instead.

```python
import plotly.graph_objects as go

fig = workflow.monitor.plot()                # 1-D fitness -> plot_obj_space_1d
assert isinstance(fig, go.Figure)
print([trace.name for trace in fig.data])    # ['Min', 'Max', 'Median', 'Average']
print(len(fig.frames))                       # 20 (animation=True is the 1-D default)

fig = workflow.monitor.plot(animation=False)  # forwarded to the builder
assert isinstance(fig, go.Figure) and not fig.frames
```

For a multi-objective run, pass the true front explicitly. The traced front of a numerical
problem is a plain function of the problem config, so build it once with `evoxc`:

```python
import evoxc
import plotly.graph_objects as go
from evox.algorithms import make_nsga2
from evox.problems.numerical.dtlz import DTLZ2, pf
from evox.workflows import EvalMonitorConfig, StdWorkflow

n_objs, d = 2, 6
problem = DTLZ2(d=d, m=n_objs, ref_num=64)      # 64 * 2 = 128 true-front points
workflow = StdWorkflow(
    make_nsga2(pop_size=40, n_objs=n_objs, lb=[0.0] * d, ub=[1.0] * d), problem,
    monitor=EvalMonitorConfig(full_fit_history=True), opt_direction=["min"] * n_objs,
    num_generations=40, backend="numpy",
)
workflow.run()

problem_pf = evoxc.run(evoxc.build(pf, problem), problem).numpy()   # (128, 2) float32
fig = workflow.monitor.plot(problem_pf=problem_pf)
assert isinstance(fig, go.Figure)
print(len(fig.frames), len(fig.frames[0].data))   # 40 generations, 2 traces (front + population)
```

The `problem_pf` array must have the same number of columns as the fitness rows; the `sort_points`
and `animation` options travel through `**kwargs` untouched.

## The EvoXVision `.exv` format

`.exv` is a small self-describing binary format for streaming optimization results — efficient
to append to, and randomly accessible per iteration.

| Magic number | Header length | Metadata | Initial iteration chunk | Binary data (rest) |
|---|---|---|---|---|
| `0x65787631` (4 bytes) | `u32` (4 bytes) | JSON, UTF-8 (`n` bytes) | binary, `chunk_size` bytes | binary chunks |

All numbers are **little-endian**. The magic `0x65787631` spells `"exv1"` (written as the bytes
`b"\x65\x78\x76\x31"`), then the `u32` metadata length, then the UTF-8 JSON metadata, then the
raw chunks. The metadata carries **two schemas** because the first iteration may legitimately
show different algorithmic behaviour (a different population size, for instance):

```json
{
  "version": "v1", "n_objs": "<number>",
  "initial_iteration": {
    "population_size": "<number>", "chunk_size": "<number>",
    "fields": [{"name": "...", "type": "...", "size": "<number>", "offset": "<number>", "shape": ["<number>"]}]
  },
  "rest_iterations": { "... same schema shape ..." }
}
```

`size` and `offset` are **bytes** within the chunk; `shape` is the logical array shape.
Supported `type` strings: `u8`, `u16`, `u32`, `u64`, `i16`, `i32`, `i64`, `f16`, `f32`, `f64`.
`bool` and `int8` are not representable — `_get_data_type` raises
`ValueError: Unsupported dtype: ...` for them.

```{note}
`.exv` files are stream-friendly: each iteration is a fixed-size chunk, so a reader can seek to
generation `g` directly (skip the header, then `initial_iteration.chunk_size +
(g - 1) * rest_iterations.chunk_size`) without decoding earlier generations.
```

## Writing `.exv` files

`new_exv_metadata(population1, population2, fitness1, fitness2)` infers the schema from the
**first two iterations** — this is why `EvoXVisionAdapter` can only start writing after two
iterations have been observed. It sets `n_objs` to `1` for a 1-D fitness array and to
`fitness1.shape[1]` otherwise, and encodes the population and fitness dtypes independently.

`EvoXVisionAdapter(file_path, buffering=0)` is the writer: `set_metadata(metadata)`,
`write_header()`, `write(*fields)` (raw `bytes`, written in order, so one chunk is
`write(pop.tobytes(), fit.tobytes())`) and `flush()`.

```python
import pathlib

from evox.algorithms import make_pso
from evox.problems.numerical import Sphere
from evox.vis_tools import EvoXVisionAdapter, new_exv_metadata
from evox.workflows import EvalMonitorConfig, StdWorkflow

run = StdWorkflow(
    make_pso(pop_size=20, lb=[-5.0] * 3, ub=[5.0] * 3), Sphere(),
    monitor=EvalMonitorConfig(full_sol_history=True),   # keep population + fitness per generation
    opt_direction="min", num_generations=4, backend="numpy",
)
run.run()

pops = run.monitor.get_solution_history()   # 4 generations of (20, 3)
fits = run.monitor.get_fitness_history()    # 4 generations of (20,)

metadata = new_exv_metadata(pops[0], pops[1], fits[0], fits[1])
path = pathlib.Path("run.exv")
adapter = EvoXVisionAdapter(path, buffering=0)
adapter.set_metadata(metadata)
adapter.write_header()
for pop, fit in zip(pops, fits):
    adapter.write(pop.tobytes(), fit.tobytes())
adapter.flush()
adapter.writer.close()
```

The adapter exposes no `close()` of its own — close the underlying handle (`adapter.writer`)
after the last chunk, since `flush()` only pushes the buffer out.

```{tip}
With `buffering=0` (the default) every `write` reaches the file immediately, so an interrupted
run still leaves a valid prefix. If you pass a non-zero `buffering`, call `flush()` explicitly —
otherwise the tail of the stream stays in the process buffer.
```

## Reading `.exv` files back

The adapter is a **writer only**: `evox` ships no reader class. Reading means parsing the
header and decoding each field with `np.frombuffer(..., offset=..., count=...)`, exactly as
`unit_test/evox/vis_tools/test_exv.py` does:

```python
import json

import numpy as np

TYPES = {"u8": np.uint8, "u16": np.uint16, "u32": np.uint32, "u64": np.uint64,
         "i16": np.int16, "i32": np.int32, "i64": np.int64,
         "f16": np.float16, "f32": np.float32, "f64": np.float64}

def decode(chunk: bytes, schema: dict) -> dict:
    return {
        f["name"]: np.frombuffer(
            chunk, dtype=TYPES[f["type"]], count=int(np.prod(f["shape"])), offset=f["offset"]
        ).reshape(f["shape"])
        for f in schema["fields"]
    }

data = path.read_bytes()
assert data[:4] == b"\x65\x78\x76\x31"                    # "exv1"
meta_len = int.from_bytes(data[4:8], "little")            # u32 metadata length
meta = json.loads(data[8 : 8 + meta_len].decode("utf-8"))
body = data[8 + meta_len :]                               # all chunks

initial, rest = meta["initial_iteration"], meta["rest_iterations"]
generations = [decode(body[: initial["chunk_size"]], initial)]
pos = initial["chunk_size"]
while pos < len(body):
    generations.append(decode(body[pos : pos + rest["chunk_size"]], rest))
    pos += rest["chunk_size"]

assert np.array_equal(generations[0]["population"], pops[0])
assert np.array_equal(generations[0]["fitness"], fits[0])
assert np.array_equal(generations[-1]["population"], pops[-1])
assert np.array_equal(generations[-1]["fitness"], fits[-1])
print(len(generations), pos == len(body))   # 4 True - every byte accounted for
```

For the run above, `meta["version"] == "v1"`, `meta["n_objs"] == 1`, both `chunk_size`s are
`320` bytes (`20 * 3 * 4` population bytes + `20 * 4` fitness bytes) and the body is
`4 * 320 = 1280` bytes; the four `assert` lines above all pass.

```{warning}
This page documents the **experimental EvoX 2.0 pre-release**. Import paths, monitored history
channels and the `.exv` metadata schema may change before the final release; the `.exv` magic
number (`v1`) is the only part that is intended to stay byte-stable.
```

Related reading: {doc}`metrics` (GD/HV/IGD on the same recorded history), {doc}`hpo`
(hyperparameter-optimization workflows that produce it), {doc}`../install/evox` (installing
`evox` and the `evox[vis]` extra) and {doc}`../../tutorial/tutorial_part6` (the guided tour).

# Choosing a CPU or a GPU

```{warning} Experimental EvoX 2.0 pre-release
These pages target the **EvoX 2.0 pre-release** (import name `evox_etl`, compiled by `evoxc`).
They do **not** cover the EvoX 1.x torch API (`import evox`, `torch.device`, `ModuleBase`,
`Parameter`/`Mutable`), which is frozen, scheduled for removal, and documented nowhere in these pages.
APIs may still change between pre-releases.
```

EvoX 2.0 has no global device setting and no environment switch to flip before `import`.
Where a run executes is decided by two keyword arguments of the `StdWorkflow` that owns the compiled
step graph: `backend=` (which compiler or interpreter executes the graph) and `device=` (which piece of
hardware). Both have deliberately dull defaults — `backend="numpy"`, `device=None`, i.e. the CPU — so a
program that never mentions either runs correctly anywhere, just not fast.

```python
import numpy as np

from evox_etl import EvalMonitorConfig, StdWorkflow
from evox_etl.algorithms import make_pso
from evox_etl.problems.numerical import Sphere

dim = 10
algorithm = make_pso(pop_size=100, lb=np.full(dim, -100.0), ub=np.full(dim, 100.0))
problem = Sphere()

workflow = StdWorkflow(
    algorithm,
    problem,
    monitor=EvalMonitorConfig(),
    backend="numpy",  # default
    device=None,      # default: CPU
)
workflow.run(50, seed=42)
```

## What `device=` accepts

`StdWorkflow` parses `device=` with `_parse_device` (`src/evox_etl/core/workflow.py`) before anything is
built, so the accepted forms are exactly these:

| `device=` | Parsed as | Rule |
|---|---|---|
| `None` | `evoxc.core.Device("cpu", 0)` | the default |
| `"cpu"`, `"cuda"`, `"rocm"`, `"vulkan"` | `Device(kind, 0)` | the string is stripped and lower-cased |
| `"cuda:1"` | `Device("cuda", 1)` | split on `:`, the index goes through `int()` |
| `evoxc.core.Device("rocm", 0)` | unchanged | passed through as an `evoxc` device |
| anything else | `TypeError` | |

The index is **0-based within the devices the process can see**, so `device="cuda:0"` is the first
visible GPU and `device="cuda:1"` the second. Parsing only splits the string: a kind that the chosen
`backend` cannot serve is rejected later, loudly, when the executable loads — never by silently falling
back to the CPU.

## Which device kinds exist

The kinds come from `evoxc`, not from EvoX: `evoxc.core.Device` describes a device as a `kind` plus an
`index`, and `evoxc.core.device` recognizes exactly four kinds.

| Kind | Meaning | Notes |
|---|---|---|
| `cpu` | the host | always available, `Device("cpu", 0)` |
| `cuda` | an NVIDIA GPU driven by the CUDA runtime | listed only when a CUDA runtime is detectable |
| `rocm` | an AMD GPU driven by ROCm/HIP | `"hip"` is accepted as an alias of the same kind |
| `vulkan` | a Vulkan device | enumerated only when asked for explicitly |

There is no `mps`, `metal`, or generic `"gpu"` kind in this pre-release — see {doc}`other_gpus` for what
that means for Apple Silicon.

## Checking what this machine offers

Ask `evoxc` instead of a GPU vendor's tooling:

```python
import evoxc

print(evoxc.devices())          # cpu plus every accelerator it can detect
print(evoxc.devices("cuda"))    # only NVIDIA devices
print(evoxc.devices("rocm"))    # only AMD devices ("hip" works too)
```

On a CPU-only host that prints:

```text
[Device(kind='cpu', index=0)]
[]
```

`evoxc.devices()` is honest and side-effect free: it never invents a device, and a kind with no runtime
present yields an empty list rather than an error.

```{note} Treat the accelerator lists as best-effort
Enumeration is a *probe*. In this pre-release the `cuda` probe reads its count from an installed
CUDA-enabled PyTorch build, and the `rocm` probe from IREE's HIP HAL driver, falling back to a ROCm
build of PyTorch. A host can therefore fail to *list* a GPU it could still serve — an empty list is a
hint, not a verdict. The verdict comes from building the workflow: a wrong `device=` fails loudly at
load time.
```

## `device=` does not create acceleration by itself

The backend has to serve the device. `backend="numpy"` is a CPU interpreter, so asking it for a GPU does
not quietly stay on the CPU — the workflow tries to place the data on that device and fails in the
transfer path. On a host with no compiled backend installed that shows up as a bare dependency error:

```text
ModuleNotFoundError: No module named 'iree'
```

The fix is to install and select a backend that serves the device (`backend="iree"` or `backend="xla"`,
with the extras and plugins described in {doc}`../guide/advanced/backends`):

```python
workflow = StdWorkflow(
    algorithm,
    problem,
    monitor=EvalMonitorConfig(),
    backend="iree",     # pip install "evox[iree]"
    device="cuda:1",    # the second visible NVIDIA GPU
)
```

## Pinning GPUs in a multi-GPU machine

Two mechanisms, and they compose:

1. **`device="cuda:N"` inside the program** — picks one device for one workflow. Each step graph runs
   on a single device; EvoX 2.0's `StdWorkflow` is single-device, with no in-graph sharding or
   cross-device collectives.
2. **`CUDA_VISIBLE_DEVICES` in the environment** — restricts what the CUDA driver exposes to the whole
   process, so indices are renumbered from zero over the remaining devices. This is a CUDA feature, not
   an EvoX one, and it applies to every CUDA consumer in the process (including IREE's CUDA HAL).

```bash
# only the second GPU is visible; inside the program it is "cuda:0"
CUDA_VISIBLE_DEVICES=1 python my_experiment.py

# two GPUs visible, for two processes (or two workflows) to use
CUDA_VISIBLE_DEVICES=0,1 python my_sweep.py

# hide the GPUs entirely: everything falls back to the CPU path you configure yourself
CUDA_VISIBLE_DEVICES="" python my_experiment.py
```

To use several GPUs, run one process per GPU and pin each one — the usual way to scale a study, since a
single workflow would not spread over them anyway:

```bash
# my_experiment.py takes the seed as argv[1] and keeps device="cuda:0"
for seed in 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=$((seed - 1)) python my_experiment.py $seed &
done
wait
```

Note that a program which *hard-codes* `device="cuda:1"` will fail under `CUDA_VISIBLE_DEVICES=1`,
because only one device is visible and its index is `0`. Either keep the index in a variable or read the
environment, and confirm what the process sees with `evoxc.devices("cuda")`.

On ROCm the equivalents are `HIP_VISIBLE_DEVICES` (and, for the ROCr runtime, `ROCR_VISIBLE_DEVICES`);
they are the standard HIP-side controls rather than EvoX arguments. Check the effect in the same
process with `evoxc.devices("rocm")` — this repository's test environment has no AMD hardware, so that
path is documented from the code and not measured here.

## Where to read next

{doc}`other_gpus` covers AMD, Vulkan and Apple Silicon hardware, and when each one is reachable;
{doc}`../guide/advanced/backends` documents the backends themselves — install extras, compile options,
the XLA PJRT plugin, and what each backend can and cannot compile.

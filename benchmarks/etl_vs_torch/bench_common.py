"""Shared infrastructure for the evox (torch) vs evox_etl benchmark harness.

IMPORTANT: this module is pure stdlib + numpy. It MUST NOT import
torch/evoxc/evox/evox_etl, because ``run.py`` imports it eagerly at the top of
``main()`` BEFORE setting the GPU environment variables (any torch/evoxc-touching
code lives in ``bench_so.py`` / ``bench_mo.py``, which are imported lazily).
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import sys
import time
from typing import Any, Callable

import numpy as np

# ---------------------------------------------------------------------------
# sys.path shim (idempotent)
# ---------------------------------------------------------------------------


def _find_repo_root() -> pathlib.Path:
    """Walk up from this file to the directory containing ``src/evox_etl/__init__.py``."""
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "src" / "evox_etl" / "__init__.py").is_file():
            return parent
    raise RuntimeError("could not locate the repo root (src/evox_etl/__init__.py)")


def bootstrap_sys_path() -> pathlib.Path:
    """Idempotently insert the repo root and ``repo_root/src`` into ``sys.path``."""
    root = _find_repo_root()
    for entry in (str(root), str(root / "src")):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    return root


REPO_ROOT = bootstrap_sys_path()  # applied at import time

HERE = pathlib.Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"

# ---------------------------------------------------------------------------
# GPU environment recipes (set BEFORE importing torch/evoxc/evox)
# ---------------------------------------------------------------------------

# --- NVIDIA (CUDA) recipe --------------------------------------------------

# cuDNN >= 9.8 needed by the xla_cuda12 PJRT plugin (the venv ships 9.1.0);
# prepended to LD_LIBRARY_PATH, keeping existing entries.
CUDNN_XLA_LIB_DIR = "/mnt/local-ssd/bchuang/cudnn-xla/lib"
XLA_FLAGS = "--xla_gpu_cuda_data_dir=/home/bchuang/xla_cuda_data"
EVOXC_PJRT_PLUGIN = (
    "/mnt/local-ssd/bchuang/evox/.venv/lib/python3.11/site-packages/"
    "jax_plugins/xla_cuda12/xla_cuda_plugin.so"
)


def set_gpu_env(gpu_id: int, preload_cudnn: bool = False) -> None:
    """Set the GPU environment recipe for in-process device id 0.

    ``CUDA_VISIBLE_DEVICES=<gpu_id>`` remaps the scanned physical GPU to device
    id 0 inside the process, so both torch (``device="cuda"``) and evoxc
    (``Device("cuda", 0)`` / ``"cuda:0"``) address it uniformly. Call this
    BEFORE importing torch/evoxc/evox modules.

    ``preload_cudnn`` additionally prepends the cuDNN >= 9.8 library to
    ``LD_PRELOAD`` (required by the xla-cuda path): the xla PJRT plugin was
    compiled against cuDNN 9.8 but carries a DT_RPATH pointing at the venv's
    pip 9.1.0 cuDNN, and DT_RPATH takes precedence over ``LD_LIBRARY_PATH``
    for the plugin's own ``dlopen("libcudnn.so.9")`` — only ``LD_PRELOAD``
    (which beats RPATH) forces the 9.8 library process-wide. Without it,
    every PJRT compile crashes with ``RET_CHECK failure
    (gpu_compiler.cc:2798) dnn_support != nullptr``. NOTE: the loader reads
    ``LD_PRELOAD`` only at process start, so this must be applied via
    ``ensure_gpu_env`` (which re-execs the process).
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if CUDNN_XLA_LIB_DIR not in ld.split(os.pathsep):
        entries = [e for e in (CUDNN_XLA_LIB_DIR, ld) if e]
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(entries)
    if preload_cudnn:
        lib = str(pathlib.Path(CUDNN_XLA_LIB_DIR) / "libcudnn.so.9")
        lp = os.environ.get("LD_PRELOAD", "")
        if lib not in lp.split():
            os.environ["LD_PRELOAD"] = " ".join(e for e in (lib, lp) if e)
    os.environ["XLA_FLAGS"] = XLA_FLAGS
    os.environ.setdefault("EVOXC_PJRT_PLUGIN", EVOXC_PJRT_PLUGIN)


# --- AMD (ROCm) recipe -----------------------------------------------------

# The xla_rocm7 PJRT plugin shipped with the ROCm benchmark venv. AMD needs
# no cuDNN preload and no ``--xla_gpu_cuda_data_dir``; HIP graph command
# buffers must be OFF because they SIGSEGV inside
# ``PJRT_Executable_Execute`` in that plugin.
ROCM_PJRT_PLUGIN = (
    "/tmp/genesis/task_3858dfe003526650/etlbench-venv/lib/python3.14/"
    "site-packages/jax_plugins/xla_rocm7/xla_rocm_plugin.so"
)
ROCM_XLA_FLAGS = "--xla_gpu_enable_command_buffer="


def set_rocm_env(gpu_id: int, enable_pjrt: bool = False) -> None:
    """Set the AMD ROCm environment recipe for in-process device id 0.

    ``HIP_VISIBLE_DEVICES`` (HIP runtime) and ``ROCR_VISIBLE_DEVICES``
    (the ROCr/HSA runtime underneath it) both remap the scanned physical GPU
    to device id 0 inside the process, so torch (``device="cuda"`` — on a
    ROCm build of torch ``torch.cuda`` IS the HIP runtime, there is no
    ``"rocm"`` torch device) and evoxc (``"rocm:0"``) address the same card.
    Call this BEFORE importing torch/evoxc/evox modules.

    ``enable_pjrt`` additionally points ``EVOXC_PJRT_PLUGIN`` at the xla_rocm7
    plugin and disables XLA/HIP graph command buffers (they SIGSEGV inside
    ``PJRT_Executable_Execute``, so every xla-rocm run must turn them off).
    An ``EVOXC_PJRT_PLUGIN`` already present in the environment wins (that is
    how another venv's plugin can be selected) and existing ``XLA_FLAGS``
    entries are preserved; setting plain strings keeps the call idempotent.
    """
    os.environ["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["ROCR_VISIBLE_DEVICES"] = str(gpu_id)
    if enable_pjrt:
        os.environ.setdefault("EVOXC_PJRT_PLUGIN", ROCM_PJRT_PLUGIN)
        flags = os.environ.get("XLA_FLAGS", "")
        if ROCM_XLA_FLAGS not in flags.split():
            os.environ["XLA_FLAGS"] = " ".join(
                e for e in (flags, ROCM_XLA_FLAGS) if e
            )


# --- shared "apply before import" machinery --------------------------------

_ENV_MARKER = "EVOX_BENCH_GPU_ENV"


def _ensure_env(recipe_key: str, apply: Callable[[], None]) -> None:
    """Apply an env recipe BEFORE any backend import, re-execing if needed.

    The dynamic loader reads ``LD_PRELOAD`` (and the CUDA/HIP drivers read the
    ``*_VISIBLE_DEVICES`` variables) at process start, so a plain in-process
    ``os.environ`` mutation is not enough: the first call re-executes this
    Python process with the recipe in the environment. ``recipe_key`` is
    stored in ``EVOX_BENCH_GPU_ENV`` so the re-exec is idempotent and callers
    may invoke this unconditionally before importing torch/evoxc (a different
    recipe key — e.g. another vendor or GPU — re-execs again).
    """
    if os.environ.get(_ENV_MARKER) == recipe_key:
        return
    apply()
    os.environ[_ENV_MARKER] = recipe_key
    os.execve(sys.executable, [sys.executable, *sys.argv], os.environ)


def ensure_gpu_env(gpu_id: int, preload_cudnn: bool = False) -> None:
    """Apply the NVIDIA CUDA env recipe (re-execs the process, idempotent)."""
    _ensure_env(
        f"nvidia:{gpu_id}:{int(bool(preload_cudnn))}",
        lambda: set_gpu_env(gpu_id, preload_cudnn),
    )


def ensure_rocm_env(gpu_id: int, enable_pjrt: bool = False) -> None:
    """Apply the AMD ROCm env recipe (re-execs the process, idempotent)."""
    _ensure_env(
        f"rocm:{gpu_id}:{int(bool(enable_pjrt))}",
        lambda: set_rocm_env(gpu_id, enable_pjrt),
    )


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

ALL_BACKENDS: tuple[str, ...] = (
    "torch-cpu",
    "torch-cuda",
    "torch-rocm",
    "etl-numpy",
    "etl-iree-llvm-cpu",
    "etl-iree-cuda",
    "etl-iree-rocm",
    "etl-xla-cuda",
    "etl-xla-rocm",
)

TORCH_BACKENDS = frozenset({"torch-cpu", "torch-cuda", "torch-rocm"})
CUDA_BACKENDS = frozenset({"torch-cuda", "etl-iree-cuda", "etl-xla-cuda"})
ROCM_BACKENDS = frozenset({"torch-rocm", "etl-iree-rocm", "etl-xla-rocm"})
# xla backends run through the PJRT client (aborts at interpreter teardown,
# see ``finish_run``)
PJRT_BACKENDS = frozenset({"etl-xla-cuda", "etl-xla-rocm"})

# vendor -> environment variables that select the physical device
GPU_VISIBLE_VARS: dict[str, tuple[str, ...]] = {
    "nvidia": ("CUDA_VISIBLE_DEVICES",),
    "rocm": ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"),
}


def is_torch_backend(backend: str) -> bool:
    return backend in TORCH_BACKENDS


def backend_vendor(backend: str) -> str | None:
    """``"nvidia"`` / ``"rocm"`` for a GPU backend, ``None`` for a CPU backend."""
    if backend in CUDA_BACKENDS:
        return "nvidia"
    if backend in ROCM_BACKENDS:
        return "rocm"
    return None


def torch_device(backend: str) -> str:
    """torch device string for a torch backend.

    A ROCm build of torch exposes HIP through ``torch.cuda`` (and
    ``torch.cuda.synchronize`` maps to ``hipDeviceSynchronize``): there is no
    ``"rocm"`` torch device, so both GPU vendors use ``"cuda"`` device 0,
    selected by ``CUDA_VISIBLE_DEVICES`` / ``HIP_VISIBLE_DEVICES``.
    """
    return "cpu" if backend == "torch-cpu" else "cuda"


def etl_backend_spec(backend: str) -> tuple[str, str | None]:
    """Map a harness backend id to ``(evoxc backend name, evoxc device spec)``.

    Returns evoxc strings only (no ``evoxc.core`` import needed here): the etl
    ``StdWorkflow`` accepts a device as ``None`` (cpu), a ``"cuda:N"`` string
    or a ``"rocm:N"`` string. The ``*_VISIBLE_DEVICES`` variable is set before
    import, so the in-process GPU device id is always 0.

    etl-xla-cuda / etl-xla-rocm: the xla adapter supports device-resident
    executables (evoxc f2f50a7+), so like the iree GPU backends they
    request ``"<vendor>:0"``. A ``None`` device would build a CPU-kind
    executable that host-stages every input on every ``evoxc.run`` call (the
    source of the old ~15 ms/step xla numbers at 10000x100 vs ~0.9-1.3
    ms/step device-resident).
    """
    if backend == "etl-numpy":
        return "numpy", None
    if backend == "etl-iree-llvm-cpu":
        return "iree", None
    if backend == "etl-iree-cuda":
        return "iree", "cuda:0"
    if backend == "etl-iree-rocm":
        return "iree", "rocm:0"
    if backend == "etl-xla-cuda":
        return "xla", "cuda:0"
    if backend == "etl-xla-rocm":
        return "xla", "rocm:0"
    raise ValueError(f"not an etl backend: {backend!r}")


def ensure_backend_env(backend: str, gpu_id: int) -> None:
    """Apply the env recipe matching ``backend``'s vendor (re-execs if needed).

    Must run BEFORE any torch/evoxc/evox import. CPU backends fall back to the
    NVIDIA recipe, which is harmless (it only pins an unused visibility
    variable) and preserves the historical behaviour of ``--gpu`` on them.
    """
    if backend_vendor(backend) == "rocm":
        ensure_rocm_env(gpu_id, enable_pjrt=(backend in PJRT_BACKENDS))
    else:
        ensure_gpu_env(gpu_id, preload_cudnn=(backend == "etl-xla-cuda"))


def gpu_selection_error(backend: str, gpu_id: int | None) -> str | None:
    """Error text when a GPU backend has no selected GPU, else ``None``."""
    vendor = backend_vendor(backend)
    if vendor is None or gpu_id is not None:
        return None
    names = GPU_VISIBLE_VARS[vendor]
    if any(name in os.environ for name in names):
        return None
    return (
        f"--backend {backend} requires --gpu N "
        f"(or {' / '.join(names)} already set in the environment)"
    )


def finish_run(backend: str) -> None:
    """Leave a completed run without PJRT interpreter teardown.

    xla (PJRT) clients abort inside the plugin's shutdown path AFTER all work
    is done and the result JSON is written, which would turn a complete run
    into a reported crash (``Aborted`` / nonzero exit). For those backends
    flush stdout/stderr and leave via ``os._exit(0)`` so a run whose records
    are complete reports success. A genuinely failed run never reaches this
    call: the exception surfaces from the case loop or ``save_results``
    (per-case failures are already written as error records). Non-PJRT
    backends simply return and exit normally.
    """
    if backend not in PJRT_BACKENDS:
        return
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


# ---------------------------------------------------------------------------
# Case tables
# ---------------------------------------------------------------------------

DEFAULT_GENS = 100
DEFAULT_SEED = 42

# Shared search-space bounds for all SO problems (Sphere/Rastrigin/Ackley);
# PSO/DE clamp to these, CMA-ES/OpenES start from the middle of the range.
SO_LB = -100.0
SO_UB = 100.0

# Deterministic ES starting points (identical across backends):
SO_ES_CENTER = 50.0  # center/mean_init value per dim (mid-way into the range)
CMAES_SIGMA = 25.0
OPENES_LR = 0.01
OPENES_NOISE_STDEV = 5.0

SO_ALGOS = ("PSO", "DE", "CMAES", "OpenES")
SO_PROBLEMS = ("Sphere", "Rastrigin", "Ackley")
SO_SCALES = ((100, 10), (1000, 50), (10000, 100))

MO_ALGOS = ("NSGA2", "NSGA3", "MOEAD")
MO_PROBLEMS = ("DTLZ1", "DTLZ2")
MO_SCALES = ((100, 3, 10), (1000, 3, 30))

MO_LB = 0.0  # DTLZ search domain is [0, 1]^d
MO_UB = 1.0


@dataclasses.dataclass(frozen=True)
class SOCase:
    algo: str
    problem: str
    pop_size: int
    dim: int
    gens: int = DEFAULT_GENS
    seed: int = DEFAULT_SEED

    @property
    def case_id(self) -> str:
        return f"{self.algo}/{self.problem}/{self.pop_size}x{self.dim}"


@dataclasses.dataclass(frozen=True)
class MOCase:
    algo: str
    problem: str
    pop_size: int
    n_obj: int
    dim: int
    gens: int = DEFAULT_GENS
    seed: int = DEFAULT_SEED

    @property
    def case_id(self) -> str:
        return f"{self.algo}/{self.problem}/{self.pop_size}x{self.n_obj}x{self.dim}"


SO_CASES: tuple[SOCase, ...] = tuple(
    SOCase(algo, problem, pop, dim)
    for algo in SO_ALGOS
    for problem in SO_PROBLEMS
    for pop, dim in SO_SCALES
)

MO_CASES: tuple[MOCase, ...] = tuple(
    MOCase(algo, problem, pop, n_obj, dim)
    for algo in MO_ALGOS
    for problem in MO_PROBLEMS
    for pop, n_obj, dim in MO_SCALES
)

CASE_TABLES: dict[str, tuple[Any, ...]] = {"so": SO_CASES, "mo": MO_CASES}


def list_case_ids(suite: str) -> list[str]:
    return [c.case_id for c in CASE_TABLES[suite]]


def select_cases(
    suite: str, case_filter: str | None, gens_override: int | None
) -> list[Any]:
    """Resolve the case list for a suite, honoring an optional comma-separated filter."""
    cases = list(CASE_TABLES[suite])
    if case_filter:
        wanted = {s.strip() for s in case_filter.split(",") if s.strip()}
        known = {c.case_id for c in cases}
        missing = sorted(wanted - known)
        if missing:
            raise SystemExit(
                f"unknown case ids: {missing}\nknown ids:\n  "
                + "\n  ".join(known)
            )
        cases = [c for c in cases if c.case_id in wanted]
    if gens_override is not None:
        cases = [dataclasses.replace(c, gens=gens_override) for c in cases]
    return cases


# ---------------------------------------------------------------------------
# Metrics (plain numpy — identical semantics for every backend, by design)
# ---------------------------------------------------------------------------

HV_NUM_SAMPLE = 100_000
HV_SEED = 0

# Reference point for hypervolume, per objective, for BOTH DTLZ1 and DTLZ2.
# Choice: DTLZ1 objectives live in [0, 1+g] with the Pareto front on the
# simplex sum(f) = 0.5; DTLZ2 objectives live in [0, 1+g] with the Pareto
# front on the unit sphere.  2.0 per objective therefore strictly dominates
# every Pareto-front point of both problems while remaining close enough to
# the running fronts that the bounding-cube Monte Carlo estimate stays
# nonzero and discriminative even during early generations.
HV_REF_VALUE = 2.0

REF_FRONT_SEED = 0
REF_FRONT_N = 1000

_front_cache: dict[tuple[str, int, int], np.ndarray] = {}


def hv(
    objs: Any,
    ref: Any,
    num_sample: int = HV_NUM_SAMPLE,
    seed: int = HV_SEED,
) -> float:
    """Monte Carlo hypervolume via the bounding-cube method.

    Identical math to ``src/evox/metrics/hv.py`` (torch version), in numpy:
    shift by |objs - ref|, bound the cube, count the fraction of uniform
    samples dominated by any point, scale by the cube volume. Computed in
    sample chunks to bound memory.
    """
    objs = np.asarray(objs, dtype=np.float64)
    if objs.ndim == 1:
        objs = objs[None, :]
    ref = np.asarray(ref, dtype=np.float64)
    if objs.shape[0] == 0:
        return 0.0
    points = np.abs(objs - ref)
    bound = np.max(points, axis=0)
    if not np.all(np.isfinite(bound)) or np.any(bound <= 0.0):
        return 0.0
    max_vol = float(np.prod(bound))
    rng = np.random.default_rng(seed)
    n, m = points.shape
    # keep the (chunk, n, m) boolean comparison under ~64 MB
    chunk = max(1, (64 * 2**20) // max(1, n * m))
    hits = 0
    for start in range(0, num_sample, chunk):
        take = min(chunk, num_sample - start)
        samples = rng.random((take, m)) * bound
        # torch: any(all(samples.unsqueeze(1) < points.unsqueeze(0), dim=2), dim=1)
        in_hypercube = np.any(
            np.all(samples[:, None, :] < points[None, :, :], axis=2), axis=1
        )
        hits += int(np.sum(in_hypercube))
    return hits / num_sample * max_vol


def igd(objs: Any, pf: Any, p: float = 1.0) -> float:
    """Inverted generational distance.

    Identical math to ``src/evox/metrics/igd.py`` (torch ``cdist`` with
    p-norm over the objective axis): the mean over reference-front points of
    the minimum distance to the solution set.
    """
    objs = np.asarray(objs, dtype=np.float64)
    if objs.ndim == 1:
        objs = objs[None, :]
    pf = np.asarray(pf, dtype=np.float64)
    if pf.ndim == 1:
        pf = pf[None, :]
    if objs.shape[0] == 0 or pf.shape[0] == 0:
        return float("inf")
    dist = np.linalg.norm(pf[:, None, :] - objs[None, :, :], ord=p, axis=2)
    min_dis = dist.min(axis=1)
    return float(np.mean(min_dis**p) ** (1.0 / p))


def dtlz_ref_front(problem: str, n_obj: int, n_ref: int = REF_FRONT_N) -> np.ndarray:
    """Deterministic reference Pareto front (fixed seed — identical everywhere).

    DTLZ1: ``n_ref`` uniform points on the simplex ``sum(f) = 0.5`` (the true
    front at g=0). DTLZ2: ``n_ref`` uniform points on the positive orthant of
    the unit sphere (the true front at g=0).
    """
    key = (problem, n_obj, n_ref)
    if key not in _front_cache:
        rng = np.random.default_rng(REF_FRONT_SEED)
        if problem == "DTLZ1":
            # uniform Dirichlet(1,...,1) samples scaled to sum 0.5
            e = rng.exponential(1.0, size=(n_ref, n_obj))
            pts = e / e.sum(axis=1, keepdims=True) * 0.5
        elif problem == "DTLZ2":
            # |N(0,1)| normalized: uniform on the positive unit-sphere orthant
            z = np.abs(rng.standard_normal((n_ref, n_obj)))
            pts = z / np.linalg.norm(z, axis=1, keepdims=True)
        else:
            raise ValueError(f"unknown DTLZ problem: {problem!r}")
        _front_cache[key] = pts.astype(np.float64)
    return _front_cache[key]


def hv_ref_point(n_obj: int) -> np.ndarray:
    return np.full(n_obj, HV_REF_VALUE, dtype=np.float64)


# ---------------------------------------------------------------------------
# Result records + JSON IO
# ---------------------------------------------------------------------------

WARMUP_STEPS = 2

# Parity tolerance vs the torch-cpu baseline (relative error). Both sides draw
# different RNG streams (torch MT19937 vs evoxc keyed PRNGs), so convergence
# parity of 10% matches the convention used by unit_test/etl/algorithms/parity.
PARITY_TOL = 0.10


def base_record(case: Any, backend: str, n_obj: int) -> dict[str, Any]:
    """Assemble the per-case JSON record with schema defaults."""
    return {
        "case_id": case.case_id,
        "backend": backend,
        "algo": case.algo,
        "problem": case.problem,
        "pop_size": case.pop_size,
        "dim": case.dim,
        "n_obj": n_obj,
        "gens": case.gens,
        "seed": case.seed,
        "compile_time_s": None,
        "warmup_steps": WARMUP_STEPS,
        "run_time_s": None,
        "ms_per_step": None,
        "metric": None,
        # "parity" is added at write time (never for the torch-cpu baseline)
        "note": None,
    }


def error_record(case: Any, backend: str, n_obj: int, exc: BaseException, note: str | None = None) -> dict[str, Any]:
    """Record for a failed case (the runner must continue after errors)."""
    rec = base_record(case, backend, n_obj)
    rec["error"] = f"{type(exc).__name__}: {exc}"
    if note:
        rec["note"] = note
    return rec


def default_results_path(suite: str, backend: str) -> pathlib.Path:
    return RESULTS_DIR / f"{suite}_{backend}.json"


def _rel_err(value: float, ref_value: float) -> float:
    denom = max(abs(ref_value), 1e-12)
    return abs(value - ref_value) / denom


def _compute_parity(suite: str, rec: dict[str, Any]) -> dict[str, Any] | None:
    """Compare a record against the torch-cpu baseline file (if it exists)."""
    baseline_path = default_results_path(suite, "torch-cpu")
    if not baseline_path.is_file():
        return None
    try:
        baseline = json.loads(baseline_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    ref = next((r for r in baseline if r.get("case_id") == rec["case_id"]), None)
    if ref is None or not isinstance(ref.get("metric"), dict):
        return None
    metric, ref_metric = rec.get("metric") or {}, ref["metric"]
    if "best_fitness" in ref_metric and "best_fitness" in metric:
        rel_err = _rel_err(metric["best_fitness"], ref_metric["best_fitness"])
        return {"vs": "torch-cpu", "rel_err": rel_err, "ok": rel_err <= PARITY_TOL}
    if "hv" in ref_metric and "hv" in metric:
        rel_hv = _rel_err(metric["hv"], ref_metric["hv"])
        rel_igd = _rel_err(metric["igd"], ref_metric["igd"])
        rel_err = max(rel_hv, rel_igd)
        return {
            "vs": "torch-cpu",
            "rel_err": rel_err,
            "rel_err_hv": rel_hv,
            "rel_err_igd": rel_igd,
            "ok": rel_hv <= PARITY_TOL and rel_igd <= PARITY_TOL,
        }
    return None


def save_results(
    suite: str,
    backend: str,
    records: list[dict[str, Any]],
    out_path: str | pathlib.Path | None = None,
) -> pathlib.Path:
    """Write the records to ``results/{suite}_{backend}.json`` (or ``--out``).

    Attaches the ``parity`` block vs the torch-cpu baseline for every
    non-torch-cpu record (skipped when the baseline file is absent). The file
    is flushed and fsynced before returning so a run that leaves through
    ``finish_run`` (PJRT teardown abort) always has complete results on disk.
    """
    out = pathlib.Path(out_path) if out_path else default_results_path(suite, backend)
    for rec in records:
        if "parity" not in rec and rec.get("backend") != "torch-cpu":
            parity = _compute_parity(suite, rec)
            if parity is not None:
                rec["parity"] = parity
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        fh.write(json.dumps(records, indent=2) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return out


def format_parity(parity: dict[str, Any] | None) -> str:
    if not parity:
        return "-"
    return f"ok={parity['ok']} rel={parity['rel_err']:.3f}"


# ---------------------------------------------------------------------------
# evoxc compile robustness
# ---------------------------------------------------------------------------

# Compiling an evoxc MO program can fail nondeterministically on GPU backends
# (upstream iree ``Stream::AffinityAnalysis`` recursion / stack exhaustion on
# the while-loop programs, plus one-off exporter rejections that do not
# reproduce on the next attempt). Rebuild + re-init the workflow a handful of
# times and keep the first SUCCESSFUL artifact. Applied to every etl backend,
# which is harmless where compiles are deterministic.
EVOXC_COMPILE_ATTEMPTS = 4
COMPILE_RETRY_BACKOFF_S = 1.0


def init_with_retry(
    build_workflow: Callable[[], Any],
    seed: int,
    label: str = "",
    attempts: int = EVOXC_COMPILE_ATTEMPTS,
) -> tuple[Any, Any, float]:
    """Build and ``init`` an etl workflow, retrying a failed compile.

    ``build_workflow`` must return a FRESH workflow on every call (a
    half-initialized one is never reused). Returns
    ``(workflow, state, compile_time_s)``, the time being the wall time of the
    attempt that succeeded. When every attempt fails the last exception is
    re-raised, so the caller records the usual per-case error record.
    ``KeyboardInterrupt`` / ``SystemExit`` are never retried.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        workflow = build_workflow()
        t0 = time.perf_counter()
        try:
            state = workflow.init(seed=seed)
        except Exception as exc:  # noqa: BLE001 — retried on the next attempt
            last_exc = exc
            print(
                f"    compile attempt {attempt}/{attempts} failed"
                f"{f' for {label}' if label else ''}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            if attempt < attempts:
                time.sleep(COMPILE_RETRY_BACKOFF_S)
            continue
        return workflow, state, time.perf_counter() - t0
    assert last_exc is not None
    raise last_exc


# ---------------------------------------------------------------------------
# small shared helpers for the runners
# ---------------------------------------------------------------------------


def run_steps(fn: Callable[[], Any], n: int) -> None:
    """Call ``fn`` ``n`` times (tiny loop helper shared by both runners)."""
    for _ in range(n):
        fn()

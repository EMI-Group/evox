"""Benchmark VirtualES against a naive OpenES implementation.

Compares **per-step wall-clock time** and **peak memory** between:

- **VirtualES**: stores only a ``(dim,)`` center vector + ``(pop_size,)`` seeds
  and generates per-individual Gaussian-noise-perturbed weights on-demand during
  the forward pass (O(dim) memory, not O(pop_size * dim)).
- **OpenES** (naive baseline): materialises a full ``(pop_size, dim)`` population.

Both algorithms optimise a *transformer-structured* model (built entirely from
``nn.Linear`` + ``nn.GELU`` layers so it remains ``VirtualProblem``-compatible)
on a synthetic classification dataset.

The benchmark sweeps population sizes from 16 up to 16384, recording time/memory
for each. When a configuration runs out of memory (CUDA OOM or CPU ``MemoryError``)
it is recorded with ``status: "oom"`` and the sweep continues. Results are saved
incrementally to JSON so partial data survives crashes, and two plots (time &
memory vs. population size) are generated at the end.

Auto-detects CUDA and falls back to CPU when unavailable.
"""

import copy
import gc
import json
import time
import tracemalloc
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from evox.algorithms import OpenES, VirtualES
from evox.core import Problem
from evox.problems.neuroevolution import VirtualProblem
from evox.problems.neuroevolution.supervised_learning import SupervisedLearningProblem
from evox.utils import ParamsAndVector
from evox.workflows import EvalMonitor, StdWorkflow

# Optional virtual-metric op. The parallel src-side additions
# (``virtual_reduce_metric`` op and ``VectorMetricProblem`` problem) may not yet
# be present in this worktree, so import them with a graceful fallback.
try:
    from evox.triton_kernels import virtual_reduce_metric
except ImportError:
    virtual_reduce_metric = None

try:
    from evox.triton_kernels.kernels.virtual_noise import _cpu_normal_noise
except ImportError:
    _cpu_normal_noise = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POP_SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
N_STEPS = 5
N_WARMUP = 2
SIGMA = 0.1
LR = 0.01
# Transformer config
D_MODEL = 512
N_LAYERS = 8
D_FF = 2048
INPUT_DIM = 512
N_CLASSES = 10
N_SAMPLES = 1024
BATCH_SIZE = 64
# Vector-metric experiment config: flat-vector length, chosen large (50M) to
# maximize memory-bandwidth pressure. Larger than the model (~29.7M) since there
# is no model-forward overhead here — it's pure memory-bandwidth.
N_PARAMS = 50_000_000


def make_transformer_model(
    d_model: int = 128,
    n_layers: int = 4,
    d_ff: int = 512,
    n_classes: int = 10,
    input_dim: int = 128,
) -> nn.Sequential:
    """Create a transformer-structured classifier built from ``nn.Linear`` + ``nn.GELU``.

    A real transformer with multi-head attention cannot be flattened into a flat
    ``nn.Sequential`` (attention requires cross-token mixing via reshapes / bmm).
    However, ``VirtualProblem`` only supports ``nn.Sequential`` whose direct
    children are ``nn.Linear`` layers and supported activation modules (it iterates
    ``model.named_children()``).

    This model therefore *mimics* transformer block structure using only
    ``nn.Linear`` + ``nn.GELU`` layers:

    - An input projection (``Linear``).
    - ``n_layers`` "transformer blocks", each containing:
      * ``Linear(d_model, 3*d_model)`` — a wide QKV-like projection.
      * ``Linear(3*d_model, d_model)`` — the output projection back to ``d_model``.
      * ``Linear(d_model, d_ff)`` / ``Linear(d_ff, d_model)`` — the feed-forward
        expand/contract pair (as in a standard transformer FFN).
    - A classification head.

    The resulting model has transformer-scale parameter count (~29.7M params with
    the default config) while remaining fully ``VirtualProblem``-compatible.

    :param d_model: Model / hidden dimension.
    :param n_layers: Number of transformer-structured blocks.
    :param d_ff: Feed-forward inner dimension.
    :param n_classes: Number of output classes.
    :param input_dim: Number of input features.
    :return: An ``nn.Sequential`` model.
    """
    layers: list[nn.Module] = [nn.Linear(input_dim, d_model)]
    for _ in range(n_layers):
        # QKV-like wide projection + output projection
        layers.append(nn.Linear(d_model, 3 * d_model))
        layers.append(nn.GELU())
        layers.append(nn.Linear(3 * d_model, d_model))
        # Feed-forward expand + contract
        layers.append(nn.Linear(d_model, d_ff))
        layers.append(nn.GELU())
        layers.append(nn.Linear(d_ff, d_model))
        layers.append(nn.GELU())
    layers.append(nn.Linear(d_model, n_classes))
    return nn.Sequential(*layers)


def make_dataset(n_samples: int, input_dim: int, n_classes: int) -> DataLoader:
    """Create a synthetic random classification dataset.

    :param n_samples: Number of samples in the dataset.
    :param input_dim: Number of input features per sample.
    :param n_classes: Number of classes.
    :return: A ``DataLoader`` (shuffled, batch size = ``BATCH_SIZE``).
    """
    x = torch.randn(n_samples, input_dim)
    y = torch.randint(0, n_classes, (n_samples,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


def benchmark_naive_es(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    pop_size: int,
    sigma: float,
    lr: float,
    n_warmup: int,
    n_steps: int,
    device: torch.device,
) -> tuple[float, int]:
    """Benchmark the naive OpenES approach.

    Sets up ``OpenES`` + ``SupervisedLearningProblem`` + ``StdWorkflow`` with
    ``solution_transform=ParamsAndVector(model)``. The full ``(pop_size, dim)``
    population is materialised and converted to a batched-params dict each step.

    Memory tracking:

    - **CUDA**: ``torch.cuda.max_memory_allocated()`` (peak reset before the timed
      loop so only timed steps are measured).
    - **CPU**: ``tracemalloc`` (started before warmup, approximate).

    :param model: The neural network model (used for param shapes / structure).
    :param data_loader: Training data loader.
    :param criterion: Loss criterion (should use ``reduction='none'``).
    :param pop_size: Population size.
    :param sigma: Noise standard deviation.
    :param lr: Learning rate.
    :param n_warmup: Number of untimed warmup steps.
    :param n_steps: Number of timed optimisation steps.
    :param device: Compute device.
    :return: ``(avg_time_per_step_seconds, peak_memory_bytes)``.
    """
    model = model.to(device)
    pv = ParamsAndVector(model)
    center_init = pv.to_vector(dict(model.named_parameters())).detach().clone()

    algo = OpenES(
        pop_size=pop_size,
        center_init=center_init,
        learning_rate=lr,
        noise_stdev=sigma,
        optimizer="adam",
        device=device,
    )
    prob = SupervisedLearningProblem(
        model=model,
        data_loader=data_loader,
        criterion=criterion,
        pop_size=pop_size,
        device=device,
        reduction="mean",
    )
    monitor = EvalMonitor(full_fit_history=True, device=device)
    workflow = StdWorkflow(
        algo,
        prob,
        monitor,
        solution_transform=ParamsAndVector(model),
        device=device,
    )

    # CPU memory tracking via tracemalloc is approximate — start before warmup.
    if device.type == "cpu":
        tracemalloc.start()

    workflow.init_step()
    for _ in range(n_warmup):
        workflow.step()

    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(n_steps):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        workflow.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)

    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated()
    else:
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mem = peak

    return sum(times) / len(times), peak_mem


def benchmark_virtual_es(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    pop_size: int,
    sigma: float,
    lr: float,
    n_warmup: int,
    n_steps: int,
    device: torch.device,
) -> tuple[float, int]:
    """Benchmark the VirtualES approach.

    Sets up ``VirtualES`` + ``VirtualProblem`` + ``StdWorkflow`` with **no**
    ``solution_transform`` — the virtual-population ``(center, seeds, sigma)`` tuple
    is passed directly to the problem for on-demand Gaussian-noise evaluation.

    Memory tracking:

    - **CUDA**: ``torch.cuda.max_memory_allocated()`` (peak reset before the timed
      loop).
    - **CPU**: ``tracemalloc`` (started before warmup, approximate).

    :param model: The neural network model (must be ``nn.Sequential``).
    :param data_loader: Training data loader.
    :param criterion: Loss criterion (should use ``reduction='none'``).
    :param pop_size: Population size.
    :param sigma: Noise standard deviation.
    :param lr: Learning rate.
    :param n_warmup: Number of untimed warmup steps.
    :param n_steps: Number of timed optimisation steps.
    :param device: Compute device.
    :return: ``(avg_time_per_step_seconds, peak_memory_bytes)``.
    """
    model = model.to(device)
    pv = ParamsAndVector(model)
    param_shapes = [tuple(p.shape) for p in model.parameters()]
    center_init = pv.to_vector(dict(model.named_parameters())).detach().clone()

    algo = VirtualES(
        param_shapes=param_shapes,
        pop_size=pop_size,
        center_init=center_init,
        learning_rate=lr,
        noise_stdev=sigma,
        optimizer="adam",
        device=device,
    )
    prob = VirtualProblem(
        model=model,
        data_loader=data_loader,
        criterion=criterion,
        n_batch_per_eval=1,
        device=device,
        reduction="mean",
    )
    monitor = EvalMonitor(full_fit_history=True, device=device)
    # NOTE: No solution_transform — the virtual path passes the (center, seeds,
    # sigma) tuple directly to the problem without conversion.
    workflow = StdWorkflow(
        algo,
        prob,
        monitor,
        device=device,
    )

    # CPU memory tracking via tracemalloc is approximate — start before warmup.
    if device.type == "cpu":
        tracemalloc.start()

    workflow.init_step()
    for _ in range(n_warmup):
        workflow.step()

    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(n_steps):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        workflow.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)

    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated()
    else:
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mem = peak

    return sum(times) / len(times), peak_mem


class VectorMetricProblem(Problem):
    """Flat-vector metric problem: fitness = ``mean(|center + sigma*noise|)`` per individual.

    Isolates the virtual-population mechanism with no model forward. Handles both
    the virtual tuple payload ``(center, seeds, sigma)`` and the naive materialized
    population ``(pop_size, n_params)``.
    """

    def __init__(self, n_params: int, device: torch.device):
        super().__init__()
        self.n_params = n_params
        self.device = device

    def evaluate(self, payload):
        if isinstance(payload, tuple):
            # Virtual path: VirtualES passes ``(center, seeds, sigma)`` where
            # ``center`` is ``(dim,)``, ``seeds`` is ``(pop_size,)`` int64, and
            # ``sigma`` is a float. The fused op returns ``(pop_size,)`` fitness
            # without ever materializing the full ``(pop_size, dim)`` population.
            center, seeds, sigma = payload
            if virtual_reduce_metric is not None:
                return virtual_reduce_metric(center, seeds, sigma, self.n_params)
            # Fallback: generate noise deterministically and compute the metric.
            #
            # NOTE: When the fused ``virtual_reduce_metric`` op is unavailable,
            # this fallback materializes the full ``(pop_size, n_params)`` noise
            # tensor — acceptable for a smoke test but NOT representative of the
            # virtual population's memory advantage. The REAL demonstration relies
            # on the fused op existing.
            if _cpu_normal_noise is not None:
                noise = _cpu_normal_noise(seeds, self.n_params, 0)
            else:
                # Last-resort fallback (only if neither is available).
                noise = torch.randn(len(seeds), self.n_params, device=center.device)
            pop = center.unsqueeze(0) + sigma * noise
            return pop.abs().mean(-1)
        else:
            # Naive path: OpenES materialises a ``(pop_size, n_params)`` population.
            return payload.abs().mean(-1)


def benchmark_naive_metric(
    n_params: int,
    pop_size: int,
    sigma: float,
    lr: float,
    n_warmup: int,
    n_steps: int,
    device: torch.device,
) -> tuple[float, int]:
    """Benchmark the naive OpenES approach on a flat-vector metric (no model).

    Uses ``OpenES`` directly on a flat ``n_params`` vector (no
    ``solution_transform``) with ``mirrored_sampling=True``. The full
    ``(pop_size, n_params)`` population is materialised each step.

    :return: ``(avg_time_per_step_seconds, peak_memory_bytes)``.
    """
    center_init = torch.randn(n_params, device=device)

    algo = OpenES(
        pop_size=pop_size,
        center_init=center_init,
        learning_rate=lr,
        noise_stdev=sigma,
        optimizer="adam",
        mirrored_sampling=True,
        device=device,
    )
    prob = VectorMetricProblem(n_params, device)
    monitor = EvalMonitor(full_fit_history=True, device=device)
    workflow = StdWorkflow(
        algo,
        prob,
        monitor,
        device=device,
    )

    # CPU memory tracking via tracemalloc is approximate — start before warmup.
    if device.type == "cpu":
        tracemalloc.start()

    workflow.init_step()
    for _ in range(n_warmup):
        workflow.step()

    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(n_steps):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        workflow.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)

    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated()
    else:
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mem = peak

    return sum(times) / len(times), peak_mem


def benchmark_virtual_metric(
    n_params: int,
    pop_size: int,
    sigma: float,
    lr: float,
    n_warmup: int,
    n_steps: int,
    device: torch.device,
) -> tuple[float, int]:
    """Benchmark the VirtualES approach on a flat-vector metric (no model).

    Uses ``VirtualES`` with a single flat block ``param_shapes=[(n_params,)]`` and
    no ``solution_transform``. The virtual ``(center, seeds, sigma)`` tuple is
    passed directly to :class:`VectorMetricProblem`.

    :return: ``(avg_time_per_step_seconds, peak_memory_bytes)``.
    """
    center_init = torch.randn(n_params, device=device)

    algo = VirtualES(
        param_shapes=[(n_params,)],
        pop_size=pop_size,
        center_init=center_init,
        learning_rate=lr,
        noise_stdev=sigma,
        optimizer="adam",
        device=device,
    )
    prob = VectorMetricProblem(n_params, device)
    monitor = EvalMonitor(full_fit_history=True, device=device)
    # NOTE: No solution_transform — the virtual path passes the (center, seeds,
    # sigma) tuple directly to the problem without conversion.
    workflow = StdWorkflow(
        algo,
        prob,
        monitor,
        device=device,
    )

    # CPU memory tracking via tracemalloc is approximate — start before warmup.
    if device.type == "cpu":
        tracemalloc.start()

    workflow.init_step()
    for _ in range(n_warmup):
        workflow.step()

    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(n_steps):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        workflow.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)

    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated()
    else:
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mem = peak

    return sum(times) / len(times), peak_mem


def _fmt_mem(n: float) -> str:
    """Format a byte count into a human-readable string."""
    if n <= 0:
        return "N/A"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _cleanup(device: torch.device) -> None:
    """Release memory between benchmark runs to avoid fragmentation."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _save_results(results_path: Path, data: dict) -> None:
    """Save (or overwrite) the full results dict to JSON."""
    with open(results_path, "w") as f:
        json.dump(data, f, indent=2)


def _successful_prefix(results: dict, key: str, pop_sizes: list[int], value_fn) -> tuple[list[int], list]:
    """Extract a contiguous prefix of successful data points for one algorithm.

    Iterates over ``pop_sizes`` in order and collects values via ``value_fn`` until
    the first entry whose ``status`` is not ``"ok"`` (e.g. OOM). The plotted line
    therefore stops at the last successful point.

    :param results: The ``"results"`` sub-dict keyed by population-size string.
    :param key: Algorithm key — ``"naive"`` or ``"virtual"``.
    :param pop_sizes: Ordered list of population sizes.
    :param value_fn: Callable ``(entry_dict) -> value`` to extract the y value.
    :return: ``(xs, ys)`` lists of population sizes and corresponding values.
    """
    xs: list[int] = []
    ys: list = []
    for ps in pop_sizes:
        entry = results.get(str(ps), {}).get(key)
        if entry is None or entry.get("status") != "ok":
            break
        xs.append(ps)
        ys.append(value_fn(entry))
    return xs, ys


def _generate_plots(
    results: dict,
    metadata: dict,
    results_dir: Path,
    series_spec: list[tuple[str, str, str]] | None = None,
    title_suffix: str = "VirtualES vs Naive OpenES",
    filename_suffix: str = "",
) -> None:
    """Generate and save the time and memory comparison plots.

    :param results: The results sub-dict keyed by population-size string.
    :param metadata: The benchmark metadata dict (used for ``pop_sizes``).
    :param results_dir: Directory to write the PNG files into.
    :param series_spec: Optional ``[(key, color, label), ...]`` overriding the
        default naive/virtual series. ``key`` indexes into each results entry.
    :param title_suffix: Title prefix used for both plot titles.
    :param filename_suffix: Appended to the output filenames (e.g. ``"_metric"``
        yields ``benchmark_metric_time.png``). Empty string keeps the originals.
    """
    if plt is None:
        print("matplotlib is not installed. Install with: pip install matplotlib. Skipping plot generation.")
        return

    pop_sizes = metadata["pop_sizes"]
    if series_spec is None:
        series = [("naive", "red", "Naive OpenES"), ("virtual", "blue", "VirtualES")]
    else:
        series = series_spec

    # --- Time plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, color, label in series:
        xs, ys = _successful_prefix(results, key, pop_sizes, lambda e: e["time_per_step"] * 1000)
        if xs:
            ax.plot(xs, ys, color=color, label=label, marker="o")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Population Size (log2)")
    ax.set_ylabel("Time per Step (ms, log)")
    ax.set_title(f"{title_suffix} — Time per Step")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    fig.tight_layout()
    time_path = results_dir / f"benchmark{filename_suffix}_time.png"
    fig.savefig(time_path)
    plt.close(fig)
    print(f"Time plot saved to {time_path}")

    # --- Memory plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, color, label in series:
        xs, ys = _successful_prefix(results, key, pop_sizes, lambda e: e["memory"] / (1024**3))
        # Guard against zero/negative memory (invalid on a log axis).
        filtered = [(x, y) for x, y in zip(xs, ys) if y > 0]
        if filtered:
            fx, fy = zip(*filtered)
            ax.plot(fx, fy, color=color, label=label, marker="o")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Population Size (log2)")
    ax.set_ylabel("Peak Memory (GB, log)")
    ax.set_title(f"{title_suffix} — Peak Memory")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    fig.tight_layout()
    mem_path = results_dir / f"benchmark{filename_suffix}_memory.png"
    fig.savefig(mem_path)
    plt.close(fig)
    print(f"Memory plot saved to {mem_path}")


def _generate_vector_metric_plots(results: dict, metadata: dict, results_dir: Path) -> None:
    """Generate time/memory plots for the flat-vector metric experiment.

    Thin wrapper around :func:`_generate_plots` using the metric-experiment
    series labels and a ``_metric`` filename suffix.
    """
    _generate_plots(
        results,
        metadata,
        results_dir,
        series_spec=[("naive", "red", "Naive OpenES (metric)"), ("virtual", "blue", "VirtualES (metric)")],
        title_suffix="VirtualES vs Naive OpenES — Vector Metric",
        filename_suffix="_metric",
    )


def plot_from_json(json_path: str | Path) -> None:
    """Regenerate the time and memory plots from a saved JSON results file.

    This allows re-plotting without re-running the (potentially long) benchmark.
    Regenerates plots for BOTH the model-based experiment (``results``) and the
    vector-metric experiment (``vector_metric_results``) when present.

    :param json_path: Path to a ``benchmark_results.json`` file produced by
        :func:`main`.
    """
    json_path = Path(json_path)
    with open(json_path) as f:
        data = json.load(f)
    results_dir = json_path.parent
    _generate_plots(data["results"], data["metadata"], results_dir)
    if "vector_metric_results" in data:
        _generate_vector_metric_plots(data["vector_metric_results"], data["metadata"], results_dir)


def main():
    """Run the population-scaling benchmark and save results + plots."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("=" * 80)

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "benchmark_results.json"

    # Build model and dataset once; deepcopy the model per run to avoid state leakage.
    model = make_transformer_model(D_MODEL, N_LAYERS, D_FF, N_CLASSES, INPUT_DIM)
    data_loader = make_dataset(N_SAMPLES, INPUT_DIM, N_CLASSES)
    criterion = nn.CrossEntropyLoss(reduction="none")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")

    all_results: dict = {
        "metadata": {
            "device": device.type,
            "timestamp": datetime.now().isoformat(),
            "model_config": {
                "d_model": D_MODEL,
                "n_layers": N_LAYERS,
                "d_ff": D_FF,
                "input_dim": INPUT_DIM,
                "n_classes": N_CLASSES,
            },
            "total_params": total_params,
            "vector_metric_config": {"n_params": N_PARAMS},
            "n_steps": N_STEPS,
            "sigma": SIGMA,
            "lr": LR,
            "pop_sizes": POP_SIZES,
        },
        "results": {},
        "vector_metric_results": {},
    }

    for pop_size in POP_SIZES:
        print(f"\n{'─' * 80}")
        print(f"Population size: {pop_size}")
        print(f"{'─' * 80}")

        pop_key = str(pop_size)
        all_results["results"][pop_key] = {}

        # --- Naive OpenES ---
        try:
            naive_time, naive_mem = benchmark_naive_es(
                model=copy.deepcopy(model),
                data_loader=data_loader,
                criterion=criterion,
                pop_size=pop_size,
                sigma=SIGMA,
                lr=LR,
                n_warmup=N_WARMUP,
                n_steps=N_STEPS,
                device=device,
            )
            all_results["results"][pop_key]["naive"] = {
                "time_per_step": naive_time,
                "memory": naive_mem,
                "status": "ok",
            }
            print(f"  [Naive OpenES]    avg time/step: {naive_time * 1000:.1f} ms | peak mem: {_fmt_mem(naive_mem)}")
        except (RuntimeError, MemoryError) as e:
            all_results["results"][pop_key]["naive"] = {"time_per_step": None, "memory": None, "status": "oom"}
            print(f"  [Naive OpenES]    OOM ({type(e).__name__}): {e}")
            _cleanup(device)

        # --- VirtualES ---
        try:
            virtual_time, virtual_mem = benchmark_virtual_es(
                model=copy.deepcopy(model),
                data_loader=data_loader,
                criterion=criterion,
                pop_size=pop_size,
                sigma=SIGMA,
                lr=LR,
                n_warmup=N_WARMUP,
                n_steps=N_STEPS,
                device=device,
            )
            all_results["results"][pop_key]["virtual"] = {
                "time_per_step": virtual_time,
                "memory": virtual_mem,
                "status": "ok",
            }
            print(f"  [VirtualES]       avg time/step: {virtual_time * 1000:.1f} ms | peak mem: {_fmt_mem(virtual_mem)}")
        except (RuntimeError, MemoryError) as e:
            all_results["results"][pop_key]["virtual"] = {"time_per_step": None, "memory": None, "status": "oom"}
            print(f"  [VirtualES]       OOM ({type(e).__name__}): {e}")
            _cleanup(device)

        # Incremental save so partial results survive crashes.
        _save_results(results_path, all_results)

        # Cleanup between population sizes to reduce memory fragmentation.
        _cleanup(device)

    # Generate plots from the completed model-based sweep.
    _generate_plots(all_results["results"], all_results["metadata"], results_dir)

    # ======================================================================
    # Second experiment: flat-vector "virtual metric" comparison.
    # Treats parameters as a flat vector and computes mean(|center + sigma*noise|)
    # per individual directly — NO model forward. This isolates the
    # virtual-population mechanism (pure memory-bandwidth).
    # ======================================================================
    print(f"\n{'=' * 80}")
    print(f"Vector-metric experiment (N_PARAMS={N_PARAMS:,})")
    print(f"{'=' * 80}")

    for pop_size in POP_SIZES:
        print(f"\n{'─' * 80}")
        print(f"Population size: {pop_size}")
        print(f"{'─' * 80}")

        pop_key = str(pop_size)
        all_results["vector_metric_results"][pop_key] = {}

        # --- Naive OpenES (metric) ---
        try:
            naive_time, naive_mem = benchmark_naive_metric(
                n_params=N_PARAMS,
                pop_size=pop_size,
                sigma=SIGMA,
                lr=LR,
                n_warmup=N_WARMUP,
                n_steps=N_STEPS,
                device=device,
            )
            all_results["vector_metric_results"][pop_key]["naive"] = {
                "time_per_step": naive_time,
                "memory": naive_mem,
                "status": "ok",
            }
            print(
                f"  [Naive metric]    avg time/step: {naive_time * 1000:.1f} ms | peak mem: {_fmt_mem(naive_mem)}"
            )
        except (RuntimeError, MemoryError) as e:
            all_results["vector_metric_results"][pop_key]["naive"] = {
                "time_per_step": None,
                "memory": None,
                "status": "oom",
            }
            print(f"  [Naive metric]    OOM ({type(e).__name__}): {e}")
            _cleanup(device)

        # --- VirtualES (metric) ---
        try:
            virtual_time, virtual_mem = benchmark_virtual_metric(
                n_params=N_PARAMS,
                pop_size=pop_size,
                sigma=SIGMA,
                lr=LR,
                n_warmup=N_WARMUP,
                n_steps=N_STEPS,
                device=device,
            )
            all_results["vector_metric_results"][pop_key]["virtual"] = {
                "time_per_step": virtual_time,
                "memory": virtual_mem,
                "status": "ok",
            }
            print(
                f"  [Virtual metric]  avg time/step: {virtual_time * 1000:.1f} ms | peak mem: {_fmt_mem(virtual_mem)}"
            )
        except (RuntimeError, MemoryError) as e:
            all_results["vector_metric_results"][pop_key]["virtual"] = {
                "time_per_step": None,
                "memory": None,
                "status": "oom",
            }
            print(f"  [Virtual metric]  OOM ({type(e).__name__}): {e}")
            _cleanup(device)

        # Incremental save so partial results survive crashes.
        _save_results(results_path, all_results)

        # Cleanup between population sizes to reduce memory fragmentation.
        _cleanup(device)

    # Generate plots for the vector-metric experiment.
    _generate_vector_metric_plots(all_results["vector_metric_results"], all_results["metadata"], results_dir)

    print(f"\n{'=' * 80}")
    print(f"Done. Results saved to {results_path}")


if __name__ == "__main__":
    main()

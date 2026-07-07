"""Benchmark VirtualLoRAES against a naive OpenES implementation.

Compares speed, memory, and convergence between:
  - **VirtualLoRAES**: stores only a (dim,) center vector + (pop_size,) seeds and
    generates per-individual LoRA-perturbed weights on-demand during the forward
    pass (O(dim) memory, not O(pop_size * dim)).
  - **OpenES** (naive baseline): materializes a full (pop_size, dim) population.

Both algorithms optimise a small MLP on a synthetic classification dataset.
Auto-detects CUDA and falls back to CPU when unavailable.
"""

import copy
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from evox.algorithms import OpenES, VirtualLoRAES
from evox.problems.neuroevolution import VirtualLoRAProblem
from evox.problems.neuroevolution.supervised_learning import SupervisedLearningProblem
from evox.utils import ParamsAndVector
from evox.workflows import EvalMonitor, StdWorkflow


def make_model(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    """Create a simple two-layer MLP classifier.

    :param input_dim: Number of input features.
    :param hidden_dim: Width of the hidden layer.
    :param output_dim: Number of output classes.
    :return: An ``nn.Sequential`` model suitable for both VirtualLoRAProblem
        (requires ``nn.Sequential``) and SupervisedLearningProblem.
    """
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def make_dataset(n_samples: int, input_dim: int, n_classes: int) -> DataLoader:
    """Create a synthetic random classification dataset.

    :param n_samples: Number of samples in the dataset.
    :param input_dim: Number of input features per sample.
    :param n_classes: Number of classes.
    :return: A ``DataLoader`` (batch_size=64, shuffle=True).
    """
    X = torch.randn(n_samples, input_dim)
    y = torch.randint(0, n_classes, (n_samples,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=64, shuffle=True)


def benchmark_naive_es(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    pop_size: int,
    sigma: float,
    lr: float,
    n_steps: int,
    device: torch.device,
) -> tuple[float, float, list[float]]:
    """Benchmark the naive OpenES approach.

    Sets up OpenES + SupervisedLearningProblem + StdWorkflow with
    ``solution_transform=ParamsAndVector(model)``.  The full ``(pop_size, dim)``
    population is materialised and converted to a batched-params dict each step.

    :param model: The neural network model (used for param shapes / structure).
    :param data_loader: Training data loader.
    :param criterion: Loss criterion (should use ``reduction='none'``).
    :param pop_size: Population size.
    :param sigma: Noise standard deviation.
    :param lr: Learning rate.
    :param n_steps: Number of optimisation steps.
    :param device: Compute device.
    :return: ``(avg_time_per_step, peak_memory_bytes, fitness_history)``.
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

    workflow.init_step()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    times = []
    fitness_history = []
    for _ in range(n_steps):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        workflow.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)
        fitness_history.append(monitor.latest_fitness.min().item())

    peak_mem = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
    return sum(times) / len(times), peak_mem, fitness_history


def benchmark_virtual_lora_es(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    pop_size: int,
    lora_rank: int,
    sigma: float,
    lr: float,
    n_steps: int,
    device: torch.device,
) -> tuple[float, float, list[float]]:
    """Benchmark the VirtualLoRAES approach.

    Sets up VirtualLoRAES + VirtualLoRAProblem + StdWorkflow with **no**
    ``solution_transform`` (the virtual-population tuple is passed directly to the
    problem for on-demand LoRA evaluation).

    :param model: The neural network model (must be ``nn.Sequential``).
    :param data_loader: Training data loader.
    :param criterion: Loss criterion (should use ``reduction='none'``).
    :param pop_size: Population size.
    :param lora_rank: LoRA rank for the low-rank perturbation factorisation.
    :param sigma: Noise standard deviation.
    :param lr: Learning rate.
    :param n_steps: Number of optimisation steps.
    :param device: Compute device.
    :return: ``(avg_time_per_step, peak_memory_bytes, fitness_history)``.
    """
    model = model.to(device)
    pv = ParamsAndVector(model)
    param_shapes = [tuple(p.shape) for p in model.parameters()]
    center_init = pv.to_vector(dict(model.named_parameters())).detach().clone()

    algo = VirtualLoRAES(
        param_shapes=param_shapes,
        lora_rank=lora_rank,
        pop_size=pop_size,
        center_init=center_init,
        learning_rate=lr,
        noise_stdev=sigma,
        optimizer="adam",
        device=device,
    )
    prob = VirtualLoRAProblem(
        model=model,
        data_loader=data_loader,
        criterion=criterion,
        lora_rank=lora_rank,
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

    workflow.init_step()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    times = []
    fitness_history = []
    for _ in range(n_steps):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        workflow.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)
        fitness_history.append(monitor.latest_fitness.min().item())

    peak_mem = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
    return sum(times) / len(times), peak_mem, fitness_history


def _fmt_mem(n: float) -> str:
    """Format a byte count into a human-readable string."""
    if n <= 0:
        return "N/A (CPU)"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main():
    """Run the benchmark across multiple model configurations and print a comparison."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("=" * 80)

    n_steps = 20
    sigma = 0.1
    lr = 0.01

    # (name, input_dim, hidden_dim, output_dim, pop_size, lora_rank)
    configs = [
        ("Small  (784→64→10)", 784, 64, 10, 64, 8),
        ("Medium (784→256→10)", 784, 256, 10, 64, 8),
        ("Large  (784→512→10)", 784, 512, 10, 64, 8),
    ]

    for name, input_dim, hidden_dim, output_dim, pop_size, lora_rank in configs:
        print(f"\n{'─' * 80}")
        print(f"Config: {name}  | pop_size={pop_size} | lora_rank={lora_rank} | steps={n_steps}")
        print(f"{'─' * 80}")

        model = make_model(input_dim, hidden_dim, output_dim)
        data_loader = make_dataset(n_samples=1000, input_dim=input_dim, n_classes=output_dim)
        criterion = nn.CrossEntropyLoss(reduction="none")

        # --- Naive OpenES ---
        naive_result = None
        try:
            naive_time, naive_mem, naive_fit = benchmark_naive_es(
                model=copy.deepcopy(model),
                data_loader=data_loader,
                criterion=criterion,
                pop_size=pop_size,
                sigma=sigma,
                lr=lr,
                n_steps=n_steps,
                device=device,
            )
            naive_result = (naive_time, naive_mem, naive_fit)
            print(f"  [Naive OpenES]    avg time/step: {naive_time * 1000:.1f} ms | "
                  f"peak mem: {_fmt_mem(naive_mem)} | final fitness: {naive_fit[-1]:.6f}")
        except Exception as e:
            print(f"  [Naive OpenES]    FAILED: {type(e).__name__}: {e}")

        # --- VirtualLoRAES ---
        virtual_result = None
        try:
            virtual_time, virtual_mem, virtual_fit = benchmark_virtual_lora_es(
                model=copy.deepcopy(model),
                data_loader=data_loader,
                criterion=criterion,
                pop_size=pop_size,
                lora_rank=lora_rank,
                sigma=sigma,
                lr=lr,
                n_steps=n_steps,
                device=device,
            )
            virtual_result = (virtual_time, virtual_mem, virtual_fit)
            print(f"  [VirtualLoRAES]   avg time/step: {virtual_time * 1000:.1f} ms | "
                  f"peak mem: {_fmt_mem(virtual_mem)} | final fitness: {virtual_fit[-1]:.6f}")
        except Exception as e:
            print(f"  [VirtualLoRAES]   FAILED: {type(e).__name__}: {e}")

        # --- Comparison ---
        print()
        if naive_result and virtual_result:
            naive_time, naive_mem, naive_fit = naive_result
            virtual_time, virtual_mem, virtual_fit = virtual_result

            speedup = naive_time / virtual_time if virtual_time > 0 else float("inf")
            print(f"  Speedup (naive / virtual):   {speedup:.2f}x")

            if device.type == "cuda" and naive_mem > 0 and virtual_mem > 0:
                mem_saved = (1 - virtual_mem / naive_mem) * 100
                print(f"  Memory saved:                {mem_saved:.1f}%")
            else:
                print(f"  Memory:                      CPU mode — no GPU memory tracking")

            delta = virtual_fit[-1] - naive_fit[-1]
            print(f"  Fitness delta (virtual-naive): {delta:+.6f}  "
                  f"(naive={naive_fit[-1]:.6f}, virtual={virtual_fit[-1]:.6f})")
        elif naive_result and not virtual_result:
            print(f"  [Comparison skipped — VirtualLoRAES failed]")
        elif virtual_result and not naive_result:
            print(f"  [Comparison skipped — Naive OpenES failed]")
        else:
            print(f"  [Comparison skipped — both failed]")

    print(f"\n{'=' * 80}")
    print("Done.")


if __name__ == "__main__":
    main()

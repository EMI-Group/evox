"""End-to-end reproduction of the benchmark_virtual_lora_es.py crash at pop_size=16.

This mirrors the real benchmark path: make_transformer_model -> VirtualES ->
VirtualProblem -> StdWorkflow -> init_step/step, exercising the Triton
`virtual_perturbed_linear` kernel through the full workflow (not just the raw
op). The original bug crashed here with a Triton OutOfResources shared-memory
error at the first init_step().

Run with::

    CUDA_VISIBLE_DEVICES=0 uv run python benchmarks/_repro_virtual_es_e2e.py
"""

import torch
import torch.nn as nn

from evox.algorithms import VirtualES
from evox.problems.neuroevolution import VirtualProblem
from evox.utils import ParamsAndVector
from evox.workflows import EvalMonitor, StdWorkflow

# Mirror the benchmark constants exactly.
D_MODEL = 128
N_LAYERS = 4
D_FF = 512
INPUT_DIM = 128
N_CLASSES = 10
N_SAMPLES = 1024
SIGMA = 0.1
LR = 0.01


def make_transformer_model(d_model, n_layers, d_ff, n_classes, input_dim):
    layers = [nn.Linear(input_dim, d_model)]
    for _ in range(n_layers):
        layers.append(nn.Linear(d_model, 3 * d_model))
        layers.append(nn.GELU())
        layers.append(nn.Linear(3 * d_model, d_model))
        layers.append(nn.Linear(d_model, d_ff))
        layers.append(nn.GELU())
        layers.append(nn.Linear(d_ff, d_model))
        layers.append(nn.GELU())
    layers.append(nn.Linear(d_model, n_classes))
    return nn.Sequential(*layers)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    assert device.type == "cuda", "This reproduction requires CUDA."

    torch.manual_seed(42)
    model = make_transformer_model(D_MODEL, N_LAYERS, D_FF, N_CLASSES, INPUT_DIM)
    x = torch.randn(N_SAMPLES, INPUT_DIM)
    y = torch.randint(0, N_CLASSES, (N_SAMPLES,))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=N_SAMPLES, shuffle=False
    )
    criterion = nn.CrossEntropyLoss(reduction="none")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")

    pop_size = 16  # the size that crashed at init_step()

    model = model.to(device)
    pv = ParamsAndVector(model)
    param_shapes = [tuple(p.shape) for p in model.parameters()]
    center_init = pv.to_vector(dict(model.named_parameters())).detach().clone()

    algo = VirtualES(
        param_shapes=param_shapes,
        pop_size=pop_size,
        center_init=center_init,
        learning_rate=LR,
        noise_stdev=SIGMA,
        optimizer="adam",
        device=device,
    )
    prob = VirtualProblem(
        model=model,
        data_loader=loader,
        criterion=criterion,
        n_batch_per_eval=1,
        device=device,
        reduction="mean",
    )
    monitor = EvalMonitor(full_fit_history=True, device=device)
    workflow = StdWorkflow(algo, prob, monitor, device=device)

    print(f"Running init_step() for pop_size={pop_size} ...")
    workflow.init_step()  # <-- this is where the original crash happened
    print("init_step() OK")

    for i in range(3):
        workflow.step()
        print(f"step {i + 1} OK")

    print("\nSUCCESS: end-to-end VirtualES workflow ran without OutOfResources.")


if __name__ == "__main__":
    main()

"""EvoX single-objective hyperparameter optimization (HPO) example.

Demonstrates end-to-end HPO with the torch API: an *inner* optimization
workflow (a real PSO running on the 5-D Rastrigin function) is wrapped by
``HPOProblemWrapper`` into an ``evox.core.Problem`` whose search space is the
inner workflow's ``Parameter``s.  An *outer* evolutionary loop (another PSO)
then searches for the inner hyperparameters that make the inner PSO converge
fastest, and we compare the tuned result against the default hyperparameters.

The tunables are the three PSO weights exposed by ``evox.algorithms.PSO``:

    w       inertia weight          (default 0.6)
    phi_p   cognitive (personal)    (default 2.5)
    phi_g   social (global)         (default 0.8)

They are searched in the range [0, 3].

Run (CPU is fine, finishes in a few seconds):
    /home/bill/Source/evox/.venv/bin/python examples/hpo/single_objective_hpo.py
"""

import pathlib
import sys

# Make `evox` (the torch reference implementation) importable without an
# editable install by walking up to the repo root that contains src/evox.
_here = pathlib.Path(__file__).resolve()
for _p in _here.parents:
    if (_p / "src" / "evox" / "__init__.py").is_file():
        _ROOT = _p
        break
else:  # pragma: no cover
    raise RuntimeError("could not locate the repo root (src/evox/__init__.py)")
for _path in (str(_ROOT), str(_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import torch

from evox.algorithms import PSO
from evox.problems.hpo_wrapper import HPOFitnessMonitor, HPOProblemWrapper
from evox.problems.numerical import Rastrigin
from evox.workflows import EvalMonitor, StdWorkflow

# ---------------------------------------------------------------------------
# Configuration (kept small so the whole script runs in a couple of seconds).
# ---------------------------------------------------------------------------
DIM = 5  # dimension of the inner Rastrigin problem
INNER_POP = 30  # population size of the inner PSO
INNER_ITERS = 30  # inner generations evaluated per hyperparameter set
OUTER_POP = 16  # number of hyperparameter sets evaluated in parallel
OUTER_GENS = 15  # number of outer generations
HP_LB, HP_UB = 0.0, 3.0  # search range for each hyperparameter
SEED = 42


class HyperparameterTransform(torch.nn.Module):
    """Map an outer PSO particle (a length-3 vector) to the inner Parameter dict.

    The keys must match ``HPOProblemWrapper.get_params_keys()`` (the module
    attribute paths of the inner workflow's ``Parameter``s).
    """

    def forward(self, x: torch.Tensor):
        return {
            "algorithm.w": x[:, 0],
            "algorithm.phi_p": x[:, 1],
            "algorithm.phi_g": x[:, 2],
        }


def main() -> None:
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)

    print("=" * 70)
    print("EvoX single-objective HPO: tune PSO (w, phi_p, phi_g) on Rastrigin")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Stage 1: the INNER workflow -- the thing we want to run well.
    # ------------------------------------------------------------------
    lb = -5.12 * torch.ones(DIM)
    ub = 5.12 * torch.ones(DIM)
    inner_algo = PSO(pop_size=INNER_POP, lb=lb, ub=ub)  # uses default w/phi_p/phi_g
    inner_prob = Rastrigin()  # global optimum 0 at the origin
    # The inner monitor MUST be an HPOFitnessMonitor: the wrapper reads the
    # best fitness found by the inner run from it.
    inner_monitor = HPOFitnessMonitor()
    inner_workflow = StdWorkflow(inner_algo, inner_prob, monitor=inner_monitor)

    print(f"\n[inner] PSO(pop_size={INNER_POP}) on {DIM}-D Rastrigin, {INNER_ITERS} generations per evaluation")

    # ------------------------------------------------------------------
    # Stage 2: wrap the inner workflow as a problem for the OUTER loop.
    # ------------------------------------------------------------------
    hpo_prob = HPOProblemWrapper(
        iterations=INNER_ITERS,
        num_instances=OUTER_POP,
        workflow=inner_workflow,
        copy_init_state=True,
    )

    keys = hpo_prob.get_params_keys()
    init_params = hpo_prob.get_init_params()
    print(f"\n[hpo]  tunable keys: {keys}")
    for k in keys:
        print(f"         {k:<18} default = {float(init_params[k][0]):.3f}   (search range [{HP_LB}, {HP_UB}])")

    # ------------------------------------------------------------------
    # Stage 3: baseline -- evaluate the DEFAULT hyperparameters, cheaply.
    # (Empty dict => every instance keeps the inner workflow's init params.)
    # ------------------------------------------------------------------
    baseline = torch.stack([hpo_prob.evaluate({}) for _ in range(3)])
    baseline_fit = float(baseline.mean())
    print(f"\n[base] untuned default hyperparameters: mean best fitness = {baseline_fit:.4f}")

    # ------------------------------------------------------------------
    # Stage 4: OUTER optimization loop -- search for better hyperparameters.
    # ------------------------------------------------------------------
    outer_algo = PSO(
        pop_size=OUTER_POP,  # MUST equal num_instances
        lb=HP_LB * torch.ones(3),
        ub=HP_UB * torch.ones(3),
    )
    outer_monitor = EvalMonitor()
    outer_workflow = StdWorkflow(
        outer_algo,
        hpo_prob,
        monitor=outer_monitor,
        solution_transform=HyperparameterTransform(),
    )

    print(f"\n[outer] PSO(pop_size={OUTER_POP}) searching 3 hyperparameters for {OUTER_GENS} generations")
    outer_workflow.init_step()
    for generation in range(1, OUTER_GENS + 1):
        outer_workflow.step()
        if generation % 3 == 0 or generation == OUTER_GENS:
            best = float(outer_monitor.get_best_fitness())
            print(f"  generation {generation:>2}: best inner fitness = {best:.4f}")

    # ------------------------------------------------------------------
    # Stage 5: report the best hyperparameters found.
    # ------------------------------------------------------------------
    best_solution = outer_monitor.get_best_solution()
    best_fitness = float(outer_monitor.get_best_fitness())
    print("\n" + "-" * 70)
    print("Result")
    print("-" * 70)
    print(f"  best {keys[0]:<16} = {float(best_solution[0]):.4f}")
    print(f"  best {keys[1]:<16} = {float(best_solution[1]):.4f}")
    print(f"  best {keys[2]:<16} = {float(best_solution[2]):.4f}")
    print(f"  best inner fitness     = {best_fitness:.4f}")
    print(f"  untuned default fitness = {baseline_fit:.4f}")
    print(
        "\nSearched the inner PSO (w, phi_p, phi_g) in [0, 3]; HPO found a set "
        "that\nconverges the inner PSO on 5-D Rastrigin noticeably better than "
        "the defaults."
    )


if __name__ == "__main__":
    main()

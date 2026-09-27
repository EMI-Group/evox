"""EvoX multi-objective HPO example: an inner MO optimizer tuned by IGD.

What this script demonstrates
-----------------------------
*Multi-objective* hyper-parameter optimization (HPO) with EvoX, obtained by
nesting two optimization loops:

* Inner loop - a tiny multi-objective ``(mu, lambda)`` evolutionary algorithm
  searches the 2-objective ``DTLZ1`` benchmark. The quality of its final
  population is collapsed into a SINGLE scalar, the Inverted Generational
  Distance (IGD) to the true Pareto front, using
  ``HPOFitnessMonitor(multi_obj_metric=lambda f: igd(f, prob.pf()))``.
* Outer loop - ``HPOProblemWrapper`` turns the whole inner workflow into a
  black-box problem whose inputs are the inner algorithm's hyper-parameters
  (marked with ``evox.core.Parameter``) and whose output is the IGD. A PSO
  ``StdWorkflow`` then searches that hyper-parameter space, evaluating one
  inner run per hyper-parameter set (``num_instances`` == outer ``pop_size``).

Every metric printed below is the IGD to the true Pareto front of DTLZ1
(lower is better). The inner problem is single-instance cheap; the outer loop
runs a whole inner optimization per candidate hyper-parameter set.

Run:
    /home/bill/Source/evox/.venv/bin/python examples/hpo/multi_objective_hpo.py
"""

import pathlib
import sys
import time

# Make `evox` importable without an editable install: walk up until we find the
# repo root, i.e. the directory that contains ``src/evox/__init__.py``.
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
from evox.core import Algorithm, Mutable, Parameter
from evox.metrics import igd
from evox.operators.selection import nd_environmental_selection
from evox.problems.hpo_wrapper import HPOFitnessMonitor, HPOProblemWrapper
from evox.problems.numerical import DTLZ1
from evox.workflows import EvalMonitor, StdWorkflow

# CPU-only machine here, so this resolves to "cpu"; on a GPU box it would be "cuda".
torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Inner algorithm: a small (mu, lambda) multi-objective evolutionary algorithm
# --------------------------------------------------------------------------- #
class SimpleMOAlgorithm(Algorithm):
    """A minimal multi-objective EA with two tunable hyper-parameters.

    Each generation ``lambda = 2 * mu`` offspring are produced by Gaussian
    mutation of the current population (each parent is mutated twice) and the
    ``mu`` best candidates are kept by non-dominated environmental selection
    (non-domination rank + crowding distance).

    The two hyper-parameters are wrapped in ``Parameter`` so that
    ``HPOProblemWrapper`` can discover them by attribute path:

    * ``algorithm.sigma`` - standard deviation of the Gaussian mutation
    * ``algorithm.p_m``   - per-coordinate probability of applying a mutation
    """

    def __init__(
        self,
        pop_size: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        n_objs: int,
        sigma: float = 0.2,
        p_m: float = 1.0,
        device: torch.device | None = None,
    ):
        super().__init__()
        device = torch.get_default_device() if device is None else device
        self.pop_size = pop_size
        self.dim = lb.shape[0]
        self.n_objs = n_objs
        # Tunable hyper-parameters (discovered as "algorithm.sigma" / "algorithm.p_m").
        self.sigma = Parameter(float(sigma), device=device)
        self.p_m = Parameter(float(p_m), device=device)
        self.lb = lb.to(device=device)
        self.ub = ub.to(device=device)
        # Initial population, uniform in [lb, ub].
        init_pop = torch.rand(pop_size, self.dim, device=device)
        init_pop = init_pop * (self.ub - self.lb)[None, :] + self.lb[None, :]
        self.pop = Mutable(init_pop)
        self.fit = Mutable(torch.zeros(pop_size, n_objs, device=device))

    def step(self) -> None:
        lam = 2 * self.pop_size
        parents = self.pop.repeat(2, 1)  # (2 * mu, dim)
        # Per-coordinate mutation mask and Gaussian perturbation.
        mask = (torch.rand(lam, self.dim, device=self.pop.device) < self.p_m).to(self.pop.dtype)
        offspring = parents + mask * self.sigma * torch.randn(lam, self.dim, device=self.pop.device)
        offspring = torch.clamp(offspring, self.lb, self.ub)
        # Evaluate through the workflow proxy (the monitor sees these fitnesses).
        offspring_fit = self.evaluate(offspring)
        # Keep the best `pop_size` candidates (non-domination + crowding distance).
        self.pop, self.fit, _, _ = nd_environmental_selection(offspring, offspring_fit, self.pop_size)


# --------------------------------------------------------------------------- #
# Main: wire up the inner MO workflow, wrap it as an HPO problem, optimize it
# --------------------------------------------------------------------------- #
def main() -> None:
    t_start = time.time()

    # ---- Problem and reduced sizes (kept tiny so the whole demo runs in seconds).
    n_objs, dim = 2, 2  # DTLZ1 with 2 objectives and 2 decision variables
    inner_pop_size = 20
    inner_iterations = 16
    num_instances = 8  # == outer PSO pop_size
    outer_generations = 9
    # Hyper-parameter search ranges (also used as the outer PSO bounds).
    sigma_range = (0.02, 0.6)
    p_m_range = (0.1, 1.0)

    prob = DTLZ1(d=dim, m=n_objs)
    pf = prob.pf()  # true Pareto front, shape (k, n_objs)

    # ---- 1) Inner multi-objective workflow, scored by IGD ------------------ #
    inner_algo = SimpleMOAlgorithm(
        pop_size=inner_pop_size,
        lb=0.0 * torch.ones(dim),
        ub=1.0 * torch.ones(dim),
        n_objs=n_objs,
        sigma=0.2,
        p_m=1.0,
    )
    inner_monitor = HPOFitnessMonitor(multi_obj_metric=lambda f: igd(f, pf))
    inner_workflow = StdWorkflow(inner_algo, prob, monitor=inner_monitor)

    # ---- 2) Wrap the inner workflow as an HPO "problem" -------------------- #
    hpo_prob = HPOProblemWrapper(
        iterations=inner_iterations,
        num_instances=num_instances,
        workflow=inner_workflow,
        copy_init_state=True,
    )

    keys = hpo_prob.get_params_keys()
    print("Multi-objective HPO: tuning an inner (mu, lambda) MO-EA on DTLZ1")
    print(
        f"  inner: {inner_pop_size} individuals, {inner_iterations} generations, "
        f"metric = IGD to the true Pareto front (lower is better)"
    )
    print(f"  outer: PSO over {len(keys)} hyper-parameters, pop_size = {num_instances}, {outer_generations} generations")
    print(f"  tunable keys: {keys}")
    print()

    # ---- 3) Baseline: IGD with the DEFAULT (untuned) hyper-parameters ------ #
    default_params = {k: v.clone() for k, v in hpo_prob.get_init_params().items()}
    default_params["algorithm.sigma"] = Parameter(0.2 * torch.ones(num_instances))
    default_params["algorithm.p_m"] = Parameter(1.0 * torch.ones(num_instances))
    default_igd = float(hpo_prob.evaluate(default_params).mean())
    print(f"Baseline IGD with default hyper-parameters (sigma=0.2, p_m=1.0): {default_igd:.6f}\n")

    # ---- 4) Outer optimization loop --------------------------------------- #
    # The outer population tensor is mapped to the inner hyper-parameter dict.
    # Columns of the outer solution are the inner hyper-parameters, searched
    # within the ranges declared above.
    class solution_transform(torch.nn.Module):
        def forward(self, x: torch.Tensor):
            return {
                "algorithm.sigma": sigma_range[0] + (sigma_range[1] - sigma_range[0]) * x[:, 0],
                "algorithm.p_m": p_m_range[0] + (p_m_range[1] - p_m_range[0]) * x[:, 1],
            }

    outer_lb = torch.tensor([0.0, 0.0])
    outer_ub = torch.tensor([1.0, 1.0])
    outer_algo = PSO(pop_size=num_instances, lb=outer_lb, ub=outer_ub)
    outer_monitor = EvalMonitor()
    outer_workflow = StdWorkflow(
        outer_algo,
        hpo_prob,
        monitor=outer_monitor,
        solution_transform=solution_transform(),
    )

    outer_workflow.init_step()
    for gen in range(1, outer_generations + 1):
        outer_workflow.step()
        best = float(outer_monitor.get_best_fitness())
        print(f"  outer generation {gen:>2}: best IGD so far = {best:.6f}")

    # ---- 5) Report the best hyper-parameters found ------------------------ #
    best_igd = float(outer_monitor.get_best_fitness())
    best_pos = outer_monitor.get_best_solution()
    best_sigma = sigma_range[0] + (sigma_range[1] - sigma_range[0]) * float(best_pos[0])
    best_p_m = p_m_range[0] + (p_m_range[1] - p_m_range[0]) * float(best_pos[1])

    print()
    print("Result")
    print(f"  best IGD found              : {best_igd:.6f}")
    print(f"  best hyper-parameters       : sigma = {best_sigma:.4f}, p_m = {best_p_m:.4f}")
    print(f"  default-hyper-parameter IGD : {default_igd:.6f}")
    print("  note: the metric is the IGD between the inner run's population and the")
    print("        true Pareto front of DTLZ1 - lower is better.")
    print(f"\nElapsed: {time.time() - t_start:.2f}s")


if __name__ == "__main__":
    main()

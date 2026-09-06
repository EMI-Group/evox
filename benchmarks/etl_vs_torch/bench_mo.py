"""Multi-objective benchmark runner: torch evox (eager) vs evox_etl (compiled).

Cases: {NSGA2, NSGA3, MOEAD} x {DTLZ1, DTLZ2} x scales {(100,3,10),
(1000,3,30)}; 100 generations, seed 42. After the run, the monitor's
whole-history Pareto front is scored with the shared numpy metric helpers
(hv with ref [2.0]*n_obj, igd vs the seeded analytic front) so every backend
is measured with IDENTICAL metric code.

Known limit: the etl iree backends are shape-risky on the NSGA-family
while-loops (cuda especially). Failures are recorded with the documented
"limit" note and the suite continues (xla-cuda is the preferred GPU MO
backend).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

import numpy as np

import bench_common  # noqa: F401  (applies the sys.path shim at import time)

MOEAD_EFFECTIVE_POP_NOTE = (
    "effective population is the Das-Dennis count (uniform_sampling overwrites "
    "the requested pop_size on both sides)"
)


def _mo_metrics(pf: Any, problem: str, n_obj: int) -> dict[str, Any]:
    """Score a Pareto front with the shared numpy metric helpers."""
    pf = np.asarray(pf, dtype=np.float64)
    ref = bench_common.hv_ref_point(n_obj)
    front = bench_common.dtlz_ref_front(problem, n_obj)
    return {
        "hv": bench_common.hv(pf, ref),
        "igd": bench_common.igd(pf, front),
        "n_pf_points": int(pf.shape[0]),
    }


# ---------------------------------------------------------------------------
# torch path (eager)
# ---------------------------------------------------------------------------


def _torch_pf_fitness(monitor: Any) -> np.ndarray:
    """Memory-safe replacement for ``monitor.get_pf_fitness()`` (torch path).

    The framework's ``get_pf_fitness`` runs ``non_dominate_rank`` over the
    whole evaluation history, which materializes ``(n, n, m)`` domination
    matrices (O(n**2) memory). At 1000-pop x 100 gens (n ~ 100k+) that needs
    ~280 GB and gets the process OOM-killed (cpu and cuda alike). The Pareto
    front of the union equals the Pareto front of the union of the
    per-generation fronts (any dominated point is dominated by a
    non-dominated point of the same generation), so this computes
    per-generation fronts (n <= pop <= ~2000, O(n**2) fine), deduplicates the
    union, and filters it incrementally in numpy (O(history x |pf|) time,
    O(|pf|) memory). Same semantics as ``get_pf_fitness(deduplicate=True)``
    under opt_direction "min".
    """
    from evox.operators.selection import non_dominate_rank

    fronts = []
    for fit in monitor.fitness_history:
        fit = fit.detach().cpu()
        rank = non_dominate_rank(fit)
        fronts.append(fit[rank == 0].numpy())
    if not fronts:
        return np.empty((0, 0))
    all_front = np.unique(np.concatenate(fronts, axis=0), axis=0)
    # incremental Pareto filter (minimization: a dominates b iff a <= b
    # elementwise and a < b in at least one objective)
    pf = np.empty((0, all_front.shape[1]), dtype=all_front.dtype)
    for row in all_front:
        if pf.shape[0] == 0:
            pf = row[None, :]
            continue
        dominated_by_pf = np.any(
            np.all(pf <= row, axis=1) & np.any(pf < row, axis=1)
        )
        if dominated_by_pf:
            continue
        keep = ~(np.all(row <= pf, axis=1) & np.any(row < pf, axis=1))
        pf = np.vstack([pf[keep], row])
    return pf


def run_torch_case(case: bench_common.MOCase, backend: str) -> dict[str, Any]:
    import torch
    from evox.algorithms import MOEAD, NSGA2, NSGA3
    from evox.problems.numerical import DTLZ1, DTLZ2
    from evox.workflows import EvalMonitor, StdWorkflow

    device = "cpu" if backend == "torch-cpu" else "cuda"
    torch.manual_seed(case.seed)

    # The torch MO algorithms allocate several tensors without an explicit
    # device: the population init in NSGA2/NSGA3/MOEAD uses the raw lb/ub
    # constructor arguments (not the device-moved copies), NSGA3's
    # vmap_get_table_row indexes with torch.arange on
    # torch.get_default_device(), and uniform_sampling builds the reference
    # points / weight vectors on the default device. Run the whole case with
    # the default device set to the target device (and lb/ub created on it)
    # so every implicit allocation lands on the same device. The torch-cpu
    # path is unaffected (its default device is cpu anyway).
    prev_default = torch.get_default_device()
    torch.set_default_device(device)
    try:
        lb = torch.full(
            (case.dim,), bench_common.MO_LB, dtype=torch.float32, device=device
        )
        ub = torch.full(
            (case.dim,), bench_common.MO_UB, dtype=torch.float32, device=device
        )
        algo_cls = {"NSGA2": NSGA2, "NSGA3": NSGA3, "MOEAD": MOEAD}[case.algo]
        algo = algo_cls(case.pop_size, case.n_obj, lb, ub, device=device)
        problem = {"DTLZ1": DTLZ1, "DTLZ2": DTLZ2}[case.problem](
            d=case.dim, m=case.n_obj
        )
        monitor = EvalMonitor(multi_obj=True, full_fit_history=True)
        workflow = StdWorkflow(
            algorithm=algo,
            problem=problem,
            monitor=monitor,
            opt_direction=["min"] * case.n_obj,
            device=device,
        )
        workflow.init_step()
        bench_common.run_steps(workflow.step, bench_common.WARMUP_STEPS)

        sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
        sync()  # drain warmup kernels before the timed window
        t0 = time.perf_counter()
        bench_common.run_steps(workflow.step, case.gens)
        sync()  # include queued kernel execution in the timed window
        run_time = time.perf_counter() - t0

        # whole-history Pareto front; _torch_pf_fitness avoids the O(n**2)
        # domination matrix that get_pf_fitness would materialize
        pf = _torch_pf_fitness(monitor)
    finally:
        torch.set_default_device(prev_default)
    rec = bench_common.base_record(case, backend, n_obj=case.n_obj)
    rec.update(
        compile_time_s=0.0,
        run_time_s=run_time,
        ms_per_step=1000.0 * run_time / case.gens,
        metric=_mo_metrics(pf, case.problem, case.n_obj),
        note="eager"
        + (f"; {MOEAD_EFFECTIVE_POP_NOTE}" if case.algo == "MOEAD" else ""),
    )
    return rec


# ---------------------------------------------------------------------------
# etl path (compiled)
# ---------------------------------------------------------------------------


def _etl_pf_fitness(monitor: Any) -> np.ndarray:
    """Memory-safe replacement for the etl monitor's ``get_pf_fitness()``.

    The etl ``EvalMonitor.get_pf_fitness`` ranks the WHOLE evaluation
    history with the O(n**2) domination matrix (``non_dominate_rank``
    materializes (n, n, m) broadcast intermediates); at 1000-pop x 100 gens
    (n ~ 100k+) each intermediate needs ~120 GB and the process is
    OOM-killed — the same limitation documented for the torch
    ``get_pf_fitness`` (see ``_torch_pf_fitness``). The Pareto front of the
    union equals the Pareto front of the union of the per-generation fronts,
    so this ranks each generation separately (n <= pop <= ~2000, O(n**2)
    fine) with the monitor's own rank operator, then deduplicates the union
    and filters it incrementally in numpy — same semantics as
    ``get_pf_fitness(deduplicate=True)`` under opt_direction "min"
    (``get_fitness_history`` undoes the opt-direction scaling exactly like
    ``get_pf_fitness`` does).
    """
    fronts = []
    for fit in monitor.get_fitness_history():
        rank = monitor._non_dominate_rank(np.asarray(fit, dtype=np.float32))
        fronts.append(np.asarray(fit)[rank == 0])
    if not fronts:
        return np.empty((0, 0))
    all_front = np.unique(np.concatenate(fronts, axis=0), axis=0)
    # incremental Pareto filter (minimization: a dominates b iff a <= b
    # elementwise and a < b in at least one objective)
    pf = np.empty((0, all_front.shape[1]), dtype=all_front.dtype)
    for row in all_front:
        if pf.shape[0] == 0:
            pf = row[None, :]
            continue
        dominated_by_pf = np.any(
            np.all(pf <= row, axis=1) & np.any(pf < row, axis=1)
        )
        if dominated_by_pf:
            continue
        keep = ~(np.all(row <= pf, axis=1) & np.any(row < pf, axis=1))
        pf = np.vstack([pf[keep], row])
    return pf


def run_etl_case(case: bench_common.MOCase, backend: str) -> dict[str, Any]:
    from evox_etl import EvalMonitorConfig, StdWorkflow
    from evox_etl.algorithms.mo.moead import make_moead
    from evox_etl.algorithms.mo.nsga2 import make_nsga2
    from evox_etl.algorithms.mo.nsga3 import make_nsga3
    from evox_etl.problems.numerical import DTLZ1, DTLZ2

    lb = np.full(case.dim, bench_common.MO_LB, np.float32)
    ub = np.full(case.dim, bench_common.MO_UB, np.float32)
    make_cfg = {"NSGA2": make_nsga2, "NSGA3": make_nsga3, "MOEAD": make_moead}[
        case.algo
    ]
    algo_cfg = make_cfg(pop_size=case.pop_size, n_objs=case.n_obj, lb=lb, ub=ub)
    problem_cfg = {"DTLZ1": DTLZ1, "DTLZ2": DTLZ2}[case.problem](
        d=case.dim, m=case.n_obj
    )
    # pop_size/dim/n_obj/multi_obj are completed by the workflow from the
    # algorithm state + the opt_direction list (MOEAD's effective Das-Dennis
    # population is discovered from the state, not the requested pop_size).
    monitor_cfg = EvalMonitorConfig()

    etl_backend, device = bench_common.etl_backend_spec(backend)
    workflow = StdWorkflow(
        algorithm=algo_cfg,
        problem=problem_cfg,
        monitor=monitor_cfg,
        opt_direction=["min"] * case.n_obj,
        num_generations=None,
        backend=etl_backend,
        device=device,
    )
    t0 = time.perf_counter()
    state = workflow.init(seed=case.seed)
    compile_time = time.perf_counter() - t0  # graph build + backend compile

    for _ in range(bench_common.WARMUP_STEPS):
        state = workflow.step(state)
    t0 = time.perf_counter()
    for _ in range(case.gens):
        state = workflow.step(state)
    run_time = time.perf_counter() - t0

    pf = _etl_pf_fitness(workflow.monitor)  # per-generation filter (see docstring)
    rec = bench_common.base_record(case, backend, n_obj=case.n_obj)
    rec.update(
        compile_time_s=compile_time,
        run_time_s=run_time,
        ms_per_step=1000.0 * run_time / case.gens,
        metric=_mo_metrics(pf, case.problem, case.n_obj),
        note=f"compiled ({etl_backend})"
        + (f"; {MOEAD_EFFECTIVE_POP_NOTE}" if case.algo == "MOEAD" else ""),
    )
    return rec


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


def _print_line(rec: dict[str, Any]) -> None:
    if rec.get("error"):
        limit = f" limit={rec['note']}" if rec.get("note") else ""
        print(f"{rec['case_id']:34s} ERROR {rec['error']}{limit}")
        return
    metric = rec.get("metric") or {}
    hv, igd = metric.get("hv"), metric.get("igd")
    hv_s = f"{hv:9.4f}" if hv is not None else "      n/a"
    igd_s = f"{igd:9.5f}" if igd is not None else "      n/a"
    parity = bench_common.format_parity(rec.get("parity"))
    print(
        f"{rec['case_id']:34s} gens={rec['gens']:>3d} "
        f"compile={rec['compile_time_s']:8.3f}s run={rec['run_time_s']:9.3f}s "
        f"ms/step={rec['ms_per_step']:9.3f} hv={hv_s} igd={igd_s} "
        f"n_pf={metric.get('n_pf_points')} parity {parity}"
    )


def run_one(case: bench_common.MOCase, backend: str) -> dict[str, Any]:
    try:
        if bench_common.is_torch_backend(backend):
            return run_torch_case(case, backend)
        return run_etl_case(case, backend)
    except BaseException as exc:  # noqa: BLE001 — record and continue
        note = None
        if backend in ("etl-iree-llvm-cpu", "etl-iree-cuda"):
            note = "documented: iree-cuda while-loop shape issue"
        return bench_common.error_record(case, backend, n_obj=case.n_obj, exc=exc, note=note)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="MO benchmarks: torch evox (eager) vs evox_etl (compiled)."
    )
    parser.add_argument(
        "--backend",
        choices=bench_common.ALL_BACKENDS,
        default="torch-cpu",
        help="backend under test (default: torch-cpu)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="physical GPU id to use (sets CUDA_VISIBLE_DEVICES; required for cuda backends)",
    )
    parser.add_argument(
        "--cases", default=None, help="comma-separated case ids to run (default: all)"
    )
    parser.add_argument("--gens-override", type=int, default=None)
    parser.add_argument(
        "--out",
        default=None,
        help="output JSON path (default: results/mo_<backend>.json)",
    )
    args = parser.parse_args(argv)

    if args.gpu is not None:
        # re-execs the process so the env recipe is active before any import
        bench_common.ensure_gpu_env(
            args.gpu, preload_cudnn=(args.backend == "etl-xla-cuda")
        )
    if (
        args.backend in bench_common.CUDA_BACKENDS
        and args.gpu is None
        and "CUDA_VISIBLE_DEVICES" not in os.environ
    ):
        parser.error(
            f"--backend {args.backend} requires --gpu N "
            "(or CUDA_VISIBLE_DEVICES already set in the environment)"
        )

    cases = bench_common.select_cases("mo", args.cases, args.gens_override)
    print(
        f"MO suite: backend={args.backend} cases={len(cases)} "
        f"gens={cases[0].gens if cases else '-'} seed={bench_common.DEFAULT_SEED}"
    )
    records = []
    for case in cases:
        print(f"  running {case.case_id} ...", flush=True)
        records.append(run_one(case, args.backend))
    out = bench_common.save_results("mo", args.backend, records, args.out)
    for rec in records:
        _print_line(rec)
    ok = sum(1 for r in records if not r.get("error"))
    print(f"wrote {len(records)} records ({ok} ok, {len(records) - ok} errors) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

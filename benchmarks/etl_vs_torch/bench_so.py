"""Single-objective benchmark runner: torch evox (eager) vs evox_etl (compiled).

Cases: {PSO, DE, CMA-ES, OpenES} x {Sphere, Rastrigin, Ackley} x
scales {(100,10), (1000,50), (10000,100)}; 100 generations, seed 42.

Torch path: ``StdWorkflow`` + ``EvalMonitor``, eager (compile_time = 0.0,
note "eager"). Etl path: ``StdWorkflow(backend=..., device=...)``; the wall
time of ``wf.init(seed=42)`` is recorded as compile time (graph build +
compile). Both sides warm up 2 steps, then time the remaining steps.

Any per-case exception is recorded as ``{"error": "<Type>: <msg>"}`` and the
runner continues (exit code 0) — errors are always printed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

import numpy as np

import bench_common  # noqa: F401  (applies the sys.path shim at import time)

# ---------------------------------------------------------------------------
# torch path (eager)
# ---------------------------------------------------------------------------


def run_torch_case(case: bench_common.SOCase, backend: str) -> dict[str, Any]:
    import torch
    from evox.algorithms import CMAES, DE, OpenES, PSO
    from evox.problems.numerical import Ackley, Rastrigin, Sphere
    from evox.workflows import EvalMonitor, StdWorkflow

    device = "cpu" if backend == "torch-cpu" else "cuda"
    torch.manual_seed(case.seed)

    lb = torch.full((case.dim,), bench_common.SO_LB, dtype=torch.float32)
    ub = torch.full((case.dim,), bench_common.SO_UB, dtype=torch.float32)
    if case.algo == "PSO":
        algo = PSO(case.pop_size, lb, ub, device=device)
    elif case.algo == "DE":
        algo = DE(case.pop_size, lb, ub, device=device)
    elif case.algo == "CMAES":
        algo = CMAES(
            torch.full((case.dim,), bench_common.SO_ES_CENTER, dtype=torch.float32),
            bench_common.CMAES_SIGMA,
            pop_size=case.pop_size,
            device=device,
        )
    else:  # OpenES
        algo = OpenES(
            case.pop_size,
            torch.full((case.dim,), bench_common.SO_ES_CENTER, dtype=torch.float32),
            bench_common.OPENES_LR,
            bench_common.OPENES_NOISE_STDEV,
            device=device,
        )
    problem = {"Sphere": Sphere, "Rastrigin": Rastrigin, "Ackley": Ackley}[
        case.problem
    ]()
    monitor = EvalMonitor(full_fit_history=True)
    workflow = StdWorkflow(
        algorithm=algo,
        problem=problem,
        monitor=monitor,
        opt_direction="min",
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

    best = float(monitor.get_best_fitness().cpu())
    rec = bench_common.base_record(case, backend, n_obj=1)
    rec.update(
        compile_time_s=0.0,
        run_time_s=run_time,
        ms_per_step=1000.0 * run_time / case.gens,
        metric={"best_fitness": best},
        note="eager",
    )
    return rec


# ---------------------------------------------------------------------------
# etl path (compiled)
# ---------------------------------------------------------------------------


def run_etl_case(case: bench_common.SOCase, backend: str) -> dict[str, Any]:
    from evox_etl import EvalMonitorConfig, StdWorkflow
    from evox_etl.algorithms.so import (
        CMAES as CMAESConfig,
        DE,
        OpenES as OpenESConfig,
        PSO,
    )
    from evox_etl.problems.numerical import Ackley, Rastrigin, Sphere

    lb = np.full(case.dim, bench_common.SO_LB, np.float32)
    ub = np.full(case.dim, bench_common.SO_UB, np.float32)
    if case.algo == "PSO":
        algo_cfg = PSO(pop_size=case.pop_size, lb=lb, ub=ub)
    elif case.algo == "DE":
        algo_cfg = DE(pop_size=case.pop_size, lb=lb, ub=ub)
    elif case.algo == "CMAES":
        algo_cfg = CMAESConfig(
            mean_init=np.full(case.dim, bench_common.SO_ES_CENTER, np.float32),
            sigma=bench_common.CMAES_SIGMA,
            pop_size=case.pop_size,
        )
    else:  # OpenES
        algo_cfg = OpenESConfig(
            pop_size=case.pop_size,
            center_init=np.full(case.dim, bench_common.SO_ES_CENTER, np.float32),
            learning_rate=bench_common.OPENES_LR,
            noise_stdev=bench_common.OPENES_NOISE_STDEV,
        )
    problem_cfg = {"Sphere": Sphere, "Rastrigin": Rastrigin, "Ackley": Ackley}[
        case.problem
    ]()
    # pop_size/dim are passed explicitly: the workflow only auto-completes the
    # monitor config from algorithm states with a `population`/`pop` attribute,
    # which CMA-ES (state `y`) and OpenES (state `noise`) do not have.
    monitor_cfg = EvalMonitorConfig(pop_size=case.pop_size, dim=case.dim)

    etl_backend, device = bench_common.etl_backend_spec(backend)
    workflow = StdWorkflow(
        algorithm=algo_cfg,
        problem=problem_cfg,
        monitor=monitor_cfg,
        opt_direction="min",
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

    best = float(np.asarray(workflow.monitor.get_best_fitness()).reshape(-1)[0])
    rec = bench_common.base_record(case, backend, n_obj=1)
    rec.update(
        compile_time_s=compile_time,
        run_time_s=run_time,
        ms_per_step=1000.0 * run_time / case.gens,
        metric={"best_fitness": best},
        note=f"compiled ({etl_backend})",
    )
    return rec


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


def _print_line(rec: dict[str, Any]) -> None:
    if rec.get("error"):
        print(f"{rec['case_id']:34s} ERROR {rec['error']}")
        return
    metric = rec.get("metric") or {}
    best = metric.get("best_fitness")
    best_s = f"{best:12.6g}" if best is not None else "        n/a"
    parity = bench_common.format_parity(rec.get("parity"))
    print(
        f"{rec['case_id']:34s} gens={rec['gens']:>3d} "
        f"compile={rec['compile_time_s']:8.3f}s run={rec['run_time_s']:9.3f}s "
        f"ms/step={rec['ms_per_step']:9.3f} best={best_s} parity {parity}"
    )


def run_one(case: bench_common.SOCase, backend: str) -> dict[str, Any]:
    try:
        if bench_common.is_torch_backend(backend):
            return run_torch_case(case, backend)
        return run_etl_case(case, backend)
    except BaseException as exc:  # noqa: BLE001 — record and continue
        return bench_common.error_record(case, backend, n_obj=1, exc=exc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SO benchmarks: torch evox (eager) vs evox_etl (compiled)."
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
        help="output JSON path (default: results/so_<backend>.json)",
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

    cases = bench_common.select_cases("so", args.cases, args.gens_override)
    print(
        f"SO suite: backend={args.backend} cases={len(cases)} "
        f"gens={cases[0].gens if cases else '-'} seed={bench_common.DEFAULT_SEED}"
    )
    records = []
    for case in cases:
        print(f"  running {case.case_id} ...", flush=True)
        records.append(run_one(case, args.backend))
    out = bench_common.save_results("so", args.backend, records, args.out)
    for rec in records:
        _print_line(rec)
    ok = sum(1 for r in records if not r.get("error"))
    print(f"wrote {len(records)} records ({ok} ok, {len(records) - ok} errors) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

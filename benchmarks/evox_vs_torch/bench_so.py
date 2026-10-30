"""Single-objective benchmark runner for the evox (evoxc) port.

Cases: {PSO, DE, CMA-ES, OpenES} x {Sphere, Rastrigin, Ackley} x
scales {(100,10), (1000,50), (10000,100)}; 100 generations, seed 42.

``StdWorkflow(backend=..., device=...)`` is built per case; the wall time of
``wf.init(seed=42)`` is recorded as compile time (graph build + compile). The
timed region is the same on every backend: first generation via
``init_step``, 2 warmup steps, then ``gens`` timed steps.

Any per-case exception is recorded as ``{"error": "<Type>: <msg>"}`` and the
runner continues (exit code 0) — errors are always printed.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import bench_common  # noqa: F401  (applies the sys.path shim at import time)
import numpy as np

# ---------------------------------------------------------------------------
# evox path (compiled)
# ---------------------------------------------------------------------------


def run_evox_case(case: bench_common.SOCase, backend: str) -> dict[str, Any]:
    from evox import EvalMonitorConfig, StdWorkflow
    from evox.algorithms.so.de_variants.de import make_de
    from evox.algorithms.so.es_variants.cma_es import make_cma_es
    from evox.algorithms.so.es_variants.open_es import make_open_es
    from evox.algorithms.so.pso_variants.pso import make_pso
    from evox.problems.numerical import Ackley, Rastrigin, Sphere

    lb = np.full(case.dim, bench_common.SO_LB, np.float32)
    ub = np.full(case.dim, bench_common.SO_UB, np.float32)
    if case.algo == "PSO":
        algo_cfg = make_pso(pop_size=case.pop_size, lb=lb, ub=ub)
    elif case.algo == "DE":
        algo_cfg = make_de(pop_size=case.pop_size, lb=lb, ub=ub)
    elif case.algo == "CMAES":
        algo_cfg = make_cma_es(
            mean_init=np.full(case.dim, bench_common.SO_ES_CENTER, np.float32),
            sigma=bench_common.CMAES_SIGMA,
            pop_size=case.pop_size,
        )
    else:  # OpenES
        algo_cfg = make_open_es(
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

    evox_backend, device = bench_common.evox_backend_spec(backend)

    def _build() -> Any:
        return StdWorkflow(
            algorithm=algo_cfg,
            problem=problem_cfg,
            monitor=monitor_cfg,
            opt_direction="min",
            num_generations=None,
            backend=evox_backend,
            device=device,
        )

    # init() is where the graph is built + compiled; a failed compile is
    # retried on a fresh workflow (see bench_common.init_with_retry).
    workflow, state, compile_time = bench_common.init_with_retry(
        _build, case.seed, label=case.case_id
    )

    # Generation 0 goes through the explicit first-generation call; the
    # remaining generations use step(). Warmup then timing structure is
    # unchanged across backends.
    state = workflow.init_step(state)
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
        note=f"compiled ({evox_backend})",
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
        return run_evox_case(case, backend)
    except BaseException as exc:  # noqa: BLE001 — record and continue
        note = None
        if case.algo == "CMAES" and type(exc).__name__ == "_LinAlgError":
            note = (
                "CMA-ES linear-algebra failure "
                "(e.g. non-PSD covariance eigendecomposition)"
            )
        return bench_common.error_record(case, backend, n_obj=1, exc=exc, note=note)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SO benchmark suite for the evox (evoxc) port."
    )
    parser.add_argument(
        "--backend",
        choices=bench_common.ALL_BACKENDS,
        required=True,
        help="backend under test (one of: " + ", ".join(bench_common.ALL_BACKENDS) + ")",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help=(
            "physical GPU id to use (sets the vendor visibility variables; "
            "required for gpu backends)"
        ),
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
        # re-execs the process so the env recipe (CUDA / ROCm) is active
        # before any evoxc import
        bench_common.ensure_backend_env(args.backend, args.gpu)
    gpu_error = bench_common.gpu_selection_error(args.backend, args.gpu)
    if gpu_error:
        parser.error(gpu_error)

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
    bench_common.finish_run(args.backend)  # PJRT: flush + os._exit (see helper)
    return 0


if __name__ == "__main__":
    sys.exit(main())

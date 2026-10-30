"""Thin CLI dispatcher for the evox_etl (evoxc) benchmark suites.

Sets the GPU environment recipe (NVIDIA CUDA or AMD ROCm, chosen from the
backend id) BEFORE importing any evoxc module (the bench runners are imported
lazily inside ``main``; ``bench_common`` is pure stdlib + numpy, so importing
it eagerly is safe).

The ``torch-*`` backends of the old torch-OOP harness are gone. The one
removed-id survivor is the parity baseline label: the committed
``results/{suite}_torch-cpu.json`` is a FROZEN HISTORICAL artefact (1.x
reference measurements) that the reporting side still compares against, but it
is not a runnable backend here.

Usage:
    run.py --list
    run.py --backend etl-numpy --suite so
    run.py --backend etl-numpy --suite mo --cases NSGA2/DTLZ2/100x3x10 --gens-override 3
    run.py --backend etl-iree-llvm-cpu --suite so
    run.py --backend etl-xla-cuda --suite so --gpu 1
    run.py --backend etl-iree-rocm --suite mo --gpu 0
    run.py --backend etl-xla-rocm --suite so --gpu 0
"""

from __future__ import annotations

import argparse
import sys

import bench_common  # noqa: F401  (sys.path shim + case tables; stdlib/numpy only)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="evox_etl (evoxc) benchmark dispatcher."
    )
    parser.add_argument(
        "--backend",
        choices=bench_common.ALL_BACKENDS,
        default=None,
        help=(
            "backend under test (not required with --list); one of: "
            + ", ".join(bench_common.ALL_BACKENDS)
        ),
    )
    parser.add_argument(
        "--suite",
        choices=["so", "mo"],
        default=None,
        help="case suite (required unless --list)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help=(
            "physical GPU id to use (sets the vendor visibility variables, "
            "CUDA or ROCm; required for gpu backends)"
        ),
    )
    parser.add_argument(
        "--cases", default=None, help="comma-separated case ids to run (default: all)"
    )
    parser.add_argument("--gens-override", type=int, default=None)
    parser.add_argument("--out", default=None, help="output JSON path override")
    parser.add_argument("--list", action="store_true", help="print all case ids and exit")
    args = parser.parse_args(argv)

    if args.list:
        for suite in ("so", "mo"):
            print(f"[{suite}]")
            for case_id in bench_common.list_case_ids(suite):
                print(f"  {case_id}")
        return 0
    if args.backend is None:
        parser.error("--backend is required (or use --list)")
    if args.suite is None:
        parser.error("--suite {so,mo} is required (or use --list)")

    # ---- GPU environment MUST be set before any evoxc import --------------
    # (ensure_backend_env re-execs the process so LD_PRELOAD / the vendor
    # visibility variables are honored by the loader/driver from process
    # start; the recipe is NVIDIA or AMD ROCm depending on the backend)
    if args.gpu is not None:
        bench_common.ensure_backend_env(args.backend, args.gpu)

    # ---- import the runner lazily (it imports evoxc inside) ---------------
    if args.suite == "so":
        import bench_so

        runner_main = bench_so.main
    else:
        import bench_mo

        runner_main = bench_mo.main

    runner_argv = ["--backend", args.backend]
    if args.cases is not None:
        runner_argv += ["--cases", args.cases]
    if args.gens_override is not None:
        runner_argv += ["--gens-override", str(args.gens_override)]
    if args.out is not None:
        runner_argv += ["--out", args.out]
    if args.gpu is not None:
        runner_argv += ["--gpu", str(args.gpu)]
    return runner_main(runner_argv)


if __name__ == "__main__":
    sys.exit(main())

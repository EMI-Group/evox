"""Merge partial benchmark runs into the committed result JSONs and re-tag
parity verdicts for the touched records, following the committed conventions
(see ``results/README.md`` and ``BENCHMARK_RESULTS.md`` §Parity verdicts).

Why this exists: the runners (``bench_so.py`` / ``bench_mo.py``) overwrite
``results/{suite}_{backend}.json`` with ONLY the records they ran, and the
parity block they attach at save time lacks the documented annotation
conventions (machine-zero rule, skip reasons, not-ok notes). Partial runs are
therefore written with ``--out <tmp>`` and merged back with this script.

Verdict rules (identical to the conventions used for the original 12 JSONs):
  * rel_err <= 0.10                          -> ok, note=null
  * both sides <= 1e-4 (SO best_fitness; MO hv AND igd)
                                             -> ok, note "both converged to
                                                machine zero"
  * record error                             -> ok=null, skip "backend error"
  * baseline (torch-cpu) record missing/error-> ok=null, skip "baseline error"
  * gens differ from baseline                -> ok=null, skip "gens capped:
                                                N vs M (parity not meaningful)"
  * otherwise                                -> not ok, note "RNG-stream
                                                variance on unconverged run"
                                                (overridable per case)

Usage:
  python3 retag_parity.py --suite so --backend etl-xla-cuda --partial /tmp/p.json
      [--cap-note "text appended to capped records' note"]
      [--note-override CASE_ID=TEXT]... [--case CASE_ID] [--apply]

Without ``--apply`` it prints the verdicts it would write (dry run).
Without ``--partial``, the touched set is ``--case`` (repeatable) or every
record in the file.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"

PARITY_TOL = 0.10
NEAR_ZERO = 1e-4

RNG_NOTE = "RNG-stream variance on unconverged run"
ZERO_NOTE = "both converged to machine zero"


def load(path: pathlib.Path) -> list[dict]:
    return json.loads(path.read_text())


def _rel_err(value: float, ref_value: float) -> float:
    return abs(value - ref_value) / max(abs(ref_value), 1e-12)


def compute_parity(suite: str, rec: dict, baseline_rec: dict | None, backend: str) -> dict:
    """Recompute the parity block for one record per the committed rules."""
    if rec.get("backend") == "torch-cpu":
        return None  # the baseline itself never carries a parity block
    if rec.get("error"):
        return {"vs": backend, "ok": None, "skip": "backend error"}
    if baseline_rec is None or baseline_rec.get("error"):
        return {"vs": backend, "ok": None, "skip": "baseline error"}
    base_metric = baseline_rec.get("metric")
    metric = rec.get("metric")
    if not isinstance(base_metric, dict) or not isinstance(metric, dict):
        return {"vs": backend, "ok": None, "skip": "baseline error"}
    if rec.get("gens") != baseline_rec.get("gens"):
        return {
            "vs": backend,
            "ok": None,
            "skip": (
                f"gens capped: {rec.get('gens')} vs {baseline_rec.get('gens')} "
                "(parity not meaningful)"
            ),
            "gens": rec.get("gens"),
            "baseline_gens": baseline_rec.get("gens"),
        }
    if suite == "so":
        a, b = metric.get("best_fitness"), base_metric.get("best_fitness")
        if a is None or b is None:
            return {"vs": backend, "ok": None, "skip": "baseline error"}
        rel = _rel_err(a, b)
        block: dict = {"vs": backend, "rel_err": rel}
        if abs(a) <= NEAR_ZERO and abs(b) <= NEAR_ZERO:
            block.update(ok=True, note=ZERO_NOTE)
        elif rel <= PARITY_TOL:
            block.update(ok=True, note=None)
        else:
            block.update(ok=False, note=RNG_NOTE)
        return block
    # mo
    hv_a, hv_b = metric.get("hv"), base_metric.get("hv")
    igd_a, igd_b = metric.get("igd"), base_metric.get("igd")
    if None in (hv_a, hv_b, igd_a, igd_b):
        return {"vs": backend, "ok": None, "skip": "baseline error"}
    rel_hv, rel_igd = _rel_err(hv_a, hv_b), _rel_err(igd_a, igd_b)
    block = {
        "vs": backend,
        "rel_err": max(rel_hv, rel_igd),
        "rel_err_hv": rel_hv,
        "rel_err_igd": rel_igd,
    }
    if all(abs(v) <= NEAR_ZERO for v in (hv_a, hv_b, igd_a, igd_b)):
        block.update(ok=True, note=ZERO_NOTE)
    elif rel_hv <= PARITY_TOL and rel_igd <= PARITY_TOL:
        block.update(ok=True, note=None)
    else:
        block.update(ok=False, note=RNG_NOTE)
    return block


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", choices=["so", "mo"], required=True)
    ap.add_argument("--backend", required=True)
    ap.add_argument("--partial", default=None, help="partial run JSON to merge")
    ap.add_argument("--case", action="append", default=[], help="case_id to retag (repeatable)")
    ap.add_argument("--cap-note", default=None, help="text appended to capped records' note")
    ap.add_argument(
        "--note-override", action="append", default=[],
        help="CASE_ID=TEXT to replace a not-ok note (repeatable)",
    )
    ap.add_argument("--apply", action="store_true", help="write the file back (default: dry run)")
    args = ap.parse_args(argv)

    path = RESULTS_DIR / f"{args.suite}_{args.backend}.json"
    committed = load(path)
    touched: set[str] = set()

    if args.partial:
        partial = load(pathlib.Path(args.partial))
        new_by_id = {r["case_id"]: r for r in partial}
        unknown = set(new_by_id) - {r["case_id"] for r in committed}
        if unknown:
            print(f"error: partial contains unknown case ids: {sorted(unknown)}", file=sys.stderr)
            return 1
        for i, r in enumerate(committed):
            if r["case_id"] in new_by_id:
                committed[i] = new_by_id[r["case_id"]]
        touched |= set(new_by_id)
    touched |= set(args.case)

    if not args.partial and not args.case:
        touched = {r["case_id"] for r in committed}

    overrides = {}
    for ov in args.note_override:
        cid, _, text = ov.partition("=")
        overrides[cid] = text

    baseline_path = RESULTS_DIR / f"{args.suite}_torch-cpu.json"
    baseline = load(baseline_path) if baseline_path.is_file() else []
    base_by_id = {r["case_id"]: r for r in baseline}

    n_retagged = 0
    for rec in committed:
        if rec["case_id"] not in touched:
            continue
        if args.cap_note and (rec.get("gens") or 100) != 100 and not rec.get("error"):
            note = rec.get("note") or ""
            if args.cap_note not in note:
                rec["note"] = f"{note}; {args.cap_note}" if note else args.cap_note
        parity = compute_parity(args.suite, rec, base_by_id.get(rec["case_id"]), "torch-cpu")
        if parity is None:
            rec.pop("parity", None)
        else:
            if parity.get("ok") is False and rec["case_id"] in overrides:
                parity["note"] = overrides[rec["case_id"]]
            rec["parity"] = parity
        n_retagged += 1
        p = rec.get("parity")
        verdict = "-" if p is None else f"ok={p['ok']} skip={p.get('skip')} rel={p.get('rel_err', p.get('rel_err', ''))}"
        print(f"{rec['case_id']:34s} -> {verdict}")

    print(f"retagged {n_retagged} of {len(committed)} records in {path.name}")
    if args.apply:
        path.write_text(json.dumps(committed, indent=2) + "\n")
        print(f"wrote {path}")
    else:
        print("dry run — pass --apply to write")
    return 0


if __name__ == "__main__":
    sys.exit(main())

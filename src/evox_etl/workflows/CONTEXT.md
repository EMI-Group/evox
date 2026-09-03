# evox_etl/workflows + metrics + utils

## Intent
`workflows/` — `std_workflow.py` (StdWorkflow, see `../core/workflow.py`) and
`eval_monitor.py` (EvalMonitor: best-so-far tracking, host-side history).
`metrics/` — igd.py, gd.py, hv.py (pure defn functions; mirror torch evox math in
`../../evox/metrics/`, read-only).
`utils/` — tree helpers, min_by, dominate_relation, pairwise_*_dist, cos_dist,
rank, rank_based_fitness, parse_opt_direction, cal_max (mirror `../../evox/utils/`).
See `../DESIGN.md`.

# evox_etl/problems — benchmark problems in functional style

## Intent
Port of the numerical problems from torch evox (`../../../evox/problems/`, read-only):
basic.py (Sphere, Rastrigin, Rosenbrock, Ackley, Griewank, Schwefel...), dtlz.py
(DTLZ1-7 + helpers), cec2022.py (with shipped input data). `evaluate(config,
problem_state, pop) -> (fitness, problem_state)` as `@etl.defn`.

NOT ported (external-library dependent — reported to root): neuroevolution problems
(brax, mujoco_playground, supervised_learning, virtual_lora), hpo_wrapper.

## Notes for Agents
- cec2022 input data lives in `../../../evox/problems/numerical/cec2022_input_data/`
  — numpy allowed ONLY for loading that data; bake it as constant tensors (closure).
- DTLZ/ZDT suites expose `pf()` reference-front generators (used by metrics/tests).

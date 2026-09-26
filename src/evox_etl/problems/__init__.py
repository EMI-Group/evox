"""Numerical benchmark problems in functional style (etl).

Port of ``src/evox/problems/numerical/`` (torch evox, read-only reference) onto
the ETL tensor library. Problems are plain functions (traced by the workflow)
with the uniform signature ``evaluate(config, problem_state, pop) ->
(fitness, problem_state)``; configs are frozen dataclasses, and all numerical
problems share the empty frozen :class:`ProblemState`.

Export surface mirrors torch evox's ``evox.problems.numerical`` plus
``Zakharov``/``Levy`` (and their plain ``*_func`` helpers, present in torch
basic.py but not re-exported there) and :class:`ProblemState`.
"""

__all__ = [
    "numerical",
    "basic",
    "cec2022",
    "dtlz",
    "ShiftAffineNumericalProblem",
    "Ackley",
    "Griewank",
    "Rastrigin",
    "Rosenbrock",
    "Schwefel",
    "Sphere",
    "Ellipsoid",
    "Zakharov",
    "Levy",
    "CEC2022",
    "DTLZ1",
    "DTLZ2",
    "DTLZ3",
    "DTLZ4",
    "DTLZ5",
    "DTLZ6",
    "DTLZ7",
    "ackley_func",
    "griewank_func",
    "rastrigin_func",
    "rosenbrock_func",
    "schwefel_func",
    "sphere_func",
    "ellipsoid_func",
    "zakharov_func",
    "levy_func",
    "ProblemState",
]

from . import numerical
from .numerical import basic, cec2022, dtlz
from .numerical.basic import (
    Ackley,
    Ellipsoid,
    Griewank,
    Levy,
    Rastrigin,
    Rosenbrock,
    Schwefel,
    ShiftAffineNumericalProblem,
    Sphere,
    Zakharov,
    ackley_func,
    ellipsoid_func,
    griewank_func,
    levy_func,
    rastrigin_func,
    rosenbrock_func,
    schwefel_func,
    sphere_func,
    zakharov_func,
)
from .numerical.cec2022 import CEC2022
from .numerical.dtlz import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7
from .numerical.state import ProblemState

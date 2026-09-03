"""evox_etl: the ETL-based functional rewrite of EvoX.

Pure functions, frozen config dataclasses, and dataclass tensor states replace
the torch OOP design; see ``DESIGN.md`` for the binding spec. Algorithms,
problems and operators stay importable as submodules (not loaded at root).
"""

__version__ = "1.3.0"

from . import core, metrics, utils, workflows

from .core import (
    Algorithm, AlgorithmConfig, AlgorithmState, Candidates, EmptyState,
    Fitness, KeyArray, Monitor, MonitorConfig, MonitorState,
    Problem, ProblemConfig, ProblemState, StdWorkflow, WorkflowState,
    get_nested, replace, set_nested,
    tree_flatten, tree_leaves, tree_map, tree_unflatten,
)
from .metrics import gd, gd_plus, hv, igd, igd_plus
from .workflows import EvalMonitor, EvalMonitorConfig

# RNG: expose etl's stateless random module and key helper
import etl.random as random
from etl.random import key

__all__ = [
    "core", "metrics", "utils", "workflows",
    "Algorithm", "AlgorithmConfig", "AlgorithmState", "Candidates",
    "EmptyState", "Fitness", "KeyArray",
    "Monitor", "MonitorConfig", "MonitorState",
    "Problem", "ProblemConfig", "ProblemState",
    "StdWorkflow", "WorkflowState",
    "get_nested", "replace", "set_nested",
    "tree_flatten", "tree_leaves", "tree_map", "tree_unflatten",
    "EvalMonitor", "EvalMonitorConfig",
    "gd", "gd_plus", "hv", "igd", "igd_plus",
    "random", "key",
]

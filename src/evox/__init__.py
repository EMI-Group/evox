"""evox: EvoX on the EvoX Compiler (2.0 functional design).

Pure functions, frozen config dataclasses and dataclass tensor states, per
``DESIGN.md`` (the binding spec). Algorithms,
problems and operators stay importable as submodules (not loaded at root).
"""

from importlib.metadata import PackageNotFoundError, version as _dist_version

try:
    __version__ = _dist_version("evox")
except PackageNotFoundError:  # running from a source tree, not an installed dist
    __version__ = "0.0.0.dev0"

from . import core, metrics, utils, vis_tools, workflows

from .core import (
    Algorithm, AlgorithmConfig, AlgorithmState, Candidates, EmptyState,
    Fitness, KeyArray, Monitor, MonitorConfig, MonitorState,
    Problem, ProblemConfig, ProblemState, StdWorkflow, WorkflowState,
    get_nested, replace, set_nested,
    tree_flatten, tree_leaves, tree_map, tree_unflatten,
)
from .metrics import gd, gd_plus, hv, igd, igd_plus
from .workflows import EvalMonitor, EvalMonitorConfig

# RNG: expose ``evoxc``'s stateless random module and key helper
import evoxc.random as random
from evoxc.random import key

__all__ = [
    "core", "metrics", "utils", "vis_tools", "workflows",
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

# Autoload the EvoX extension namespace (guarded: tolerate its absence).
try:
    from evox_ext.autoload_ext import auto_load_extensions

    auto_load_extensions()
except ImportError:
    pass

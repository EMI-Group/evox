"""Core of evox_etl: state helpers, functional protocols, and the standard workflow."""

from .algorithm import (
    Algorithm,
    AlgorithmConfig,
    AlgorithmState,
    Candidates,
    Fitness,
    KeyArray,
)
from .monitor import Monitor, MonitorConfig, MonitorState
from .problem import Problem, ProblemConfig, ProblemState
from .state import (
    get_nested,
    replace,
    set_nested,
    tree_flatten,
    tree_leaves,
    tree_map,
    tree_unflatten,
)
from .workflow import EmptyState, StdWorkflow, WorkflowState

__all__ = [
    "Algorithm",
    "AlgorithmConfig",
    "AlgorithmState",
    "Candidates",
    "Fitness",
    "KeyArray",
    "Monitor",
    "MonitorConfig",
    "MonitorState",
    "Problem",
    "ProblemConfig",
    "ProblemState",
    "get_nested",
    "replace",
    "set_nested",
    "tree_map",
    "tree_leaves",
    "tree_flatten",
    "tree_unflatten",
    "EmptyState",
    "WorkflowState",
    "StdWorkflow",
]

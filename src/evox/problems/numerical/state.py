"""Problem state for evox numerical problems."""

import dataclasses

__all__ = ["ProblemState"]


@dataclasses.dataclass(frozen=True)
class ProblemState:
    """Stateless problem state shared by all numerical problems."""

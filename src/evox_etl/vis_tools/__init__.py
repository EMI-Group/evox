"""evox_etl.vis_tools — host-side visualization utilities for the ETL EvoX rewrite.

Interactive Plotly figure builders (a near-verbatim functional port of the torch
``evox.vis_tools`` package) plus the EvoXVision (``.exv``) binary serialization
format.  Plotly is an OPTIONAL dependency — importing this package never requires
it, only calling one of the ``plot_*`` functions does.
"""

from .exv import EvoXVisionAdapter, new_exv_metadata
from .plot import (
    plot_dec_space,
    plot_obj_space_1d,
    plot_obj_space_1d_animation,
    plot_obj_space_1d_no_animation,
    plot_obj_space_2d,
    plot_obj_space_3d,
)

__all__ = [
    "plot_dec_space",
    "plot_obj_space_1d",
    "plot_obj_space_1d_animation",
    "plot_obj_space_1d_no_animation",
    "plot_obj_space_2d",
    "plot_obj_space_3d",
    "new_exv_metadata",
    "EvoXVisionAdapter",
]

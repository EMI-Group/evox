__all__ = [
    # modules
    "core",
    "utils",
    "algorithms",
    "problems",
    "workflows",
    "operators",
    "vis_tools",
    "metrics",
    # re-exported classes and functions
    "ModuleBase",
    "Mutable",
    "Parameter",
    "compile",
    "use_state",
    "vmap",
]

from . import core
# re-export everything from core, so that users can access it directly
from .core import ModuleBase, Mutable, Parameter, compile, use_state, vmap
from . import algorithms, metrics, operators, problems, utils, vis_tools, workflows


# After that, try loading extensions from `evox_ext` package
from evox_ext.autoload_ext import auto_load_extensions

auto_load_extensions()

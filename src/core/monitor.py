from abc import ABC
from typing import Any

from .module import *


class Monitor(ModuleBase, ABC):
    """
    The monitor base class.

    Monitors are used to monitor the evolutionary process.
    They contains a set of callbacks,
    which will be called at specific points during the execution of the workflow.
    Monitor itself lives outside the main workflow, so jit is not required.

    To implements a monitor, implement your own callbacks and override the hooks method.
    The hooks method should return a list of strings, which are the names of the callbacks.
    Currently the supported callbacks are:

    `pre_step`, `post_step`, `pre_ask`, `post_ask`, `pre_eval`, `post_eval`, `pre_tell`, `post_tell`, and `post_step`.
    """
    
    def __init__(self):
        raise NotImplementedError()
    
    def set_opt_direction(self, opt_direction):
        raise NotImplementedError()
    
    def hooks(self):
        raise NotImplementedError
    
    def pre_step(self):
        raise NotImplementedError()
    
    def pre_ask(self):
        raise NotImplementedError()
    
    def post_ask(self, cand_sol: Union[torch.Tensor, Any]):
        raise NotImplementedError()
    
    def pre_eval(self, cand_sol: Union[torch.Tensor, Any], transformed_cand_sol: Union[torch.Tensor, Any]):
        raise NotImplementedError()
    
    def post_eval(self, cand_sol: Union[torch.Tensor, Any], transformed_cand_sol: Union[torch.Tensor, Any],
                  fitness: torch.Tensor):
        raise NotImplementedError()
    
    def pre_tell(self, cand_sol: Union[torch.Tensor, Any], transformed_cand_sol: Union[torch.Tensor, Any],
                 fitness: torch.Tensor, transformed_fitness: torch.Tensor):
        raise NotImplementedError()
    
    def post_tell(self):
        raise NotImplementedError()
    
    def post_step(self):
        raise NotImplementedError()

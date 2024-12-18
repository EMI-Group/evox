import sys
from typing import Dict

# sys.path.append(__file__ + "/../..")

import torch
from torch import nn
from core import Algorithm, Problem, Workflow, jit_class


@jit_class
class HpoProblemWrapper(Problem):
    
    def __init__(self, iterations: int):
        super().__init__()
        self.iterations = iterations
        
    def setup(self, workflow: Workflow):
        self.workflow = workflow
        
    
    def evaluate(self, hyper_parameters: Dict[str, nn.Parameter]):
        
        self.workflow.init_step()
        for _ in range(self.iterations):
            self.workflow.step()
        return self.workflow.solution
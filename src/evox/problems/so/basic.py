import torch

from ...core import Problem, jit_class

def ackley_func(a:float, b:float, c:float, x: torch.Tensor) -> torch.Tensor:
        return (
        -a * torch.exp(-b * torch.sqrt(torch.mean(x**2, dim=1)))
        - torch.exp(torch.mean(torch.cos(c * x), dim=1))
        + a
        + torch.e
    )

@jit_class
class Ackley(Problem):
    def __init__(self, a:float = 20.0 , b:float = 0.2, c:float = 2*torch.pi):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        return ackley_func(self.a, self.b, self.c, x)

def griewank_func(x: torch.Tensor) -> torch.Tensor:
    f = (
        1 / 4000 * torch.sum(x**2, dim=1)
        - torch.prod(torch.cos(x / torch.sqrt(torch.arange(1, x.size(1) + 1))))
        + 1
    )
    return f

@jit_class
class Griewank(Problem):
    def evaluate(self, x):
        return griewank_func(x)
    

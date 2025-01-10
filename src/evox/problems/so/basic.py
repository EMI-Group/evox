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
    

def rastrigin_func(x: torch.Tensor) -> torch.Tensor:
    return 10 * x.size(1) + torch.sum(x**2 - 10 * torch.cos(2 * torch.pi * x), dim=1)

@jit_class
class Rastrigin(Problem):
    def evaluate(self,x):
        return rastrigin_func(x)
 
    
def rosenbrock_func(x):
    f = torch.sum(
        100 * ((x[:,1:])- x[:,:x.size(1) - 1] ** 2) ** 2 + (x[:,:x.size(1) - 1] - 1) ** 2,
        dim=1)
    return f

@jit_class
class Rosenbrock(Problem):
    def evaluate(self, x):
        return rosenbrock_func(x)
    

def schwefel_func(x):
    return (418.9828872724338 * x.size(1) 
   - torch.sum(x * torch.sin(torch.sqrt(torch.abs(x))), dim=1))


@jit_class
class Schwefel(Problem):
    """The Schwefel function
    The minimum is x = [420.9687462275036, ...]
    """
    def evaluate(self, x):
        return schwefel_func(x)
    

def sphere_func(X):
    return torch.sum(x**2, axis=1)


@jit_class
class Sphere(Problem):
    def evaluate(self, x):
        return sphere_func(x)
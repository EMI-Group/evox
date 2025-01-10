import torch

from ...core import Problem, jit_class

def ackley_func(a:float, b:float, c:float, x: torch.Tensor) -> torch.Tensor:
        return (
        -a * torch.exp(-b * torch.sqrt(torch.mean(x**2, dim=1)))
        - torch.exp(torch.mean(torch.cos(c * x), dim=1))
        + a
        + torch.e)
@jit_class
class Ackley(Problem):
    def __init__(self, a:float = 20.0 , b:float = 0.2, c:float = 2*torch.pi):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        return ackley_func(self.a, self.b, self.c, x)


def bent_cigar_func(x: torch.Tensor) -> torch.Tensor:
    return x[:, 0] ** 2 + torch.sum((10.0**6) * x[:, 1:] ** 2, dim=1)
@jit_class
class BentCigar(Problem):
    def evaluate(self, x):
        return bent_cigar_func(x)


def discus_func(x: torch.Tensor) -> torch.Tensor:
    return (10.0**6) * x[:, 0] ** 2 + torch.sum(x[:, 1:] ** 2, dim=1)
@jit_class
class Discus(Problem):
    def evaluate(self, x):
        return discus_func(x)
    

def ellips_func(x: torch.Tensor) -> torch.Tensor:
    idx = torch.arange(x.size(1), device=x.device)
    powers = 6.0 * idx / (x.size(1) - 1)
    return torch.sum((10.0**powers) * x**2, dim=1)
@jit_class
class Ellips(Problem):
    def evaluate(self, x):
        return ellips_func(x)


def griewank_func(x: torch.Tensor) -> torch.Tensor:
    return (1 / 4000 * torch.sum(x**2, dim=1)
        - torch.prod(torch.cos(x / torch.sqrt(torch.arange(1, x.size(1) + 1))))
        + 1)
@jit_class
class Griewank(Problem):
    def evaluate(self, x):
        return griewank_func(x)


def happycat_func(x: torch.Tensor) -> torch.Tensor:
    alpha = 1.0 / 8.0
    nx = x.size(1)
    tmp = x - 1
    r2 = torch.sum(tmp**2, dim=1)
    sum_x = torch.sum(tmp, dim=1)
    return torch.abs(r2 - nx) ** (2 * alpha) + (0.5 * r2 + sum_x) / nx + 0.5
@jit_class
class Happycat(Problem):
    def evaluate(self, x):
        return happycat_func(x)


def hgbat_func(x: torch.Tensor) -> torch.Tensor:
    alpha = 1.0 / 4.0
    tmp = x - 1
    r2 = torch.sum(tmp**2, dim=1)
    sum_x = torch.sum(tmp, dim=1)
    return torch.abs(r2**2 - sum_x**2) ** (2 * alpha) + (0.5 * r2 + sum_x) / x.size(1) + 0.5
@jit_class
class Hgbat(Problem):
    def evaluate(self, x):
        return hgbat_func(x)


def katsuura_func(x: torch.Tensor) -> torch.Tensor:
    nx = x.size(1)
    tmp1 = 2.0 ** torch.arange(1, 33, device=x.device)
    tmp2 = x.unsqueeze(-1) * tmp1.unsqueeze(0).unsqueeze(0)
    temp = torch.sum(torch.abs(tmp2 - torch.floor(tmp2 + 0.5)) / tmp1, dim=2)
    tmp3 = torch.arange(1, nx + 1, device=x.device)
    f = torch.prod((1 + temp * tmp3.unsqueeze(0)) ** (10.0 / (nx**1.2)), dim=1)
    return (f - 1) * (10.0 / nx / nx)
@jit_class
class Katsuura(Problem):
    def evaluate(self, x):
        return katsuura_func(x)
    

def levy_func(x: torch.Tensor) -> torch.Tensor:
    w = 1.0 + x / 4.0
    tmp1 = torch.sin(torch.pi * w[:, 0]) ** 2
    tmp2 = (w[:, -1] - 1) ** 2 * (1 + torch.sin(2 * torch.pi * w[:, -1]) ** 2)
    sum = (w[:, :-1] - 1) ** 2 * (1 + 10 * torch.sin(torch.pi * w[:, :-1] + 1) ** 2)
    return torch.sum(tmp1.unsqueeze(1) + sum + tmp2.unsqueeze(1), dim=1)
@jit_class
class Levy(Problem):
    def evaluate(self, x):
        return levy_func(x)
    

def rastrigin_func(x: torch.Tensor) -> torch.Tensor:
    return 10 * x.size(1) + torch.sum(x**2 - 10 * torch.cos(2 * torch.pi * x), dim=1)
@jit_class
class Rastrigin(Problem):
    def evaluate(self,x):
        return rastrigin_func(x)
 
    
def rosenbrock_func(x: torch.Tensor) -> torch.Tensor:
    f = torch.sum(
        100 * ((x[:,1:])- x[:,:x.size(1) - 1] ** 2) ** 2 + (x[:,:x.size(1) - 1] - 1) ** 2,
        dim=1)
    return f
@jit_class
class Rosenbrock(Problem):
    def evaluate(self, x):
        return rosenbrock_func(x)


def schaffer_F7_func(x: torch.Tensor) -> torch.Tensor:
    tmp1 = torch.hypot(x[:, :-1], x[:, 1:])
    tmp2 = torch.sin(50.0 * x[:, :-1] ** 0.2)
    f = torch.sqrt(tmp1) * (1 + tmp2 * tmp2)
    f = torch.mean(f, dim=1)
    return f * f
@jit_class
class SchafferF7(Problem):
    def evaluate(self, x):
        return schaffer_F7_func(x)


def schwefel_func(x: torch.Tensor) -> torch.Tensor:
    return (418.9828872724338 * x.size(1) 
   - torch.sum(x * torch.sin(torch.sqrt(torch.abs(x))), dim=1))
@jit_class
class Schwefel(Problem):
    """The Schwefel function
    The minimum is x = [420.9687462275036, ...]
    """
    def evaluate(self, x):
        return schwefel_func(x)
    

def sphere_func(x: torch.Tensor) -> torch.Tensor:
    return (x**2).sum(-1)
@jit_class
class Sphere(Problem):
    def evaluate(self, x):
        return sphere_func(x)


def zakharov_func(x: torch.Tensor) -> torch.Tensor:
    sum1 = x**2
    sum2 = 0.5 * (torch.arange(1, x.size(1) + 1, device=x.device) * x)
    return torch.sum(sum1 + sum2**2 + sum2**4, dim=1)
@jit_class
class Zakharov(Problem):
    def evaluate(self, x):
        return zakharov_func(x)



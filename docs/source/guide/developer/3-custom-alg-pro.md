# Custom algorithms and problems in EvoX

In this chapter, we will introduce how to implement your own algorithms and problems in EvoX.

## Layout of the algorithms and problems

In most traditional EC libraries, algorithms usually call the objective function internally, which gives the following layout:

```
Algorithm
|
+--Problem
```

**But in EvoX, we have a flat layout:**

```
Algorithm.step -- Problem.evaluate
```

This layout makes both algorithms and problems more universal: an algorithm can optimize different problems, while a problem can also be suitable for many algorithms.



## Algorithm class

The {class}`Algorithm <evox.Algorithm>` class is inherited from {class}`ModuleBase <evox.core.ModuleBase>`.

**In total,** **there are 5 methods (2 methods are optional) that we need to implement:**

| Method       | Signature                               | Usage                                                                                                              |
| ------------ | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `__init__` | `(self, ...)`                   | Initialize hyperparameters that are fixed though out the optimization process, for example, the `population size`, the `low bound` and the `up bound`. |
| `setup` (optional) | `(self, *args, **kwargs) -> self` | Make the mutable initialization of the algorithm.  Most of the time there's no need to override it. |
| `step`               | `(self)`                        | Perform a normal optimization iteration step of the algorithm. |
| `init_step` (optional) | `(self)` | Perform the first step of the optimization of the algorithm.  If not override this method, the first step will automatically use the `step` method. |

#### Notice

The static initialization can still be written in the `__init__` while the mutable initialization cannot. Therefore, multiple calls of `setup` for multiple initializations are possible.

If such `setup` method in {class}`ModuleBase <evox.core.ModuleBase>` is not suitable for your algorithm, you can override the `setup` method when you create your own algorithm class.



## Problem class

The {class}`Problem <evox.Problem>` class is also inherited from {class}`ModuleBase <evox.core.ModuleBase>`. 

However, the Problem class is quite simpler. **Beside the`__init__` method, the only necessary method is the `evaluate` method.**

| Method     | Signature                                   | Usage                                         |
| ---------- | ------------------------------------------- | --------------------------------------------- |
| `__init__` | `(self, ...)`                       | Initialize the settings of the problem.       |
| `evaluate` | `(self, pop:torch.Tensor) -> Array` | Evaluate the fitness of the given population. |



## Example

Here we give an example of **implementing a PSO algorithm that solves the Sphere problem**.

### Pseudo-code of the example

Here is a pseudo-code:

```text
Set hyperparameters

Generate the initial population
Do
    Compute fitness
    
    Update the lbest and the gbest
    Update the velocity
    Udpdate the population
    
Until stopping criterion
```

And here is what each part of the algorithm and the problem corresponds to in EvoX.

```python
Set hyperparameters # Algortihm.__init__

Generate the initial population # Algortihm.setup
Do
    # Problem.evaluate (not part of the algorithm)
    Compute fitness
    
    # Algortihm.step
    Update the lbest and the gbest
    Update the velocity
    Udpdate the population

Until stopping criterion
```

### Algorithm example: PSO algorithm

Particle Swarm Optimization (PSO) is a population-based metaheuristic algorithm inspired by the social behavior of birds and fish. It is widely used for solving continuous and discrete optimization problems.

**Here is an implementation example of PSO algorithm in EvoX:**

```python
import torch
from typing import List

from evox.utils import clamp
from evox.core import Parameter, Mutable, Algorithm, jit_class

@jit_class
class PSO(Algorithm):
    #Initialize the PSO algorithm with the given parameters.
    def __init__(
        self,
        pop_size: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        w: float = 0.6,
        phi_p: float = 2.5,
        phi_g: float = 0.8,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert lb.shape == ub.shape and lb.ndim == 1 and ub.ndim == 1 and lb.dtype == ub.dtype
        self.pop_size = pop_size
        self.dim = lb.shape[0]
        # Here, Parameter is used to indicate that these values are hyper-parameters
        # so that they can be correctly traced and vector-mapped
        self.w = Parameter(w, device=device)
        self.phi_p = Parameter(phi_p, device=device)
        self.phi_g = Parameter(phi_g, device=device)
        lb = lb[None, :].to(device=device)
        ub = ub[None, :].to(device=device)
        length = ub - lb
        population = torch.rand(self.pop_size, self.dim, device=device)
        population = length * population + lb
        velocity = torch.rand(self.pop_size, self.dim, device=device)
        velocity = 2 * length * velocity - length
        self.lb = lb
        self.ub = ub
        # Mutable parameters
        self.population = Mutable(population)
        self.velocity = Mutable(velocity)
        self.local_best_location = Mutable(population)
        self.local_best_fitness = Mutable(torch.empty(self.pop_size, device=device).fill_(torch.inf))
        self.global_best_location = Mutable(population[0])
        self.global_best_fitness = Mutable(torch.tensor(torch.inf, device=device))

    def step(self):
        # Compute fitness
        fitness = self.evaluate(self.population)
        
        # Update the lbest and the gbest
        compare = self.local_best_fitness - fitness
        self.local_best_location = torch.where(
            compare[:, None] > 0, self.population, self.local_best_location
        )
        self.local_best_fitness = self.local_best_fitness - torch.relu(compare)
        self.global_best_location, self.global_best_fitness = self._min_by(
            [self.global_best_location.unsqueeze(0), self.population],
            [self.global_best_fitness.unsqueeze(0), fitness],
        )
        
        # Update the velocity
        rg = torch.rand(self.pop_size, self.dim, dtype=fitness.dtype, device=fitness.device)
        rp = torch.rand(self.pop_size, self.dim, dtype=fitness.dtype, device=fitness.device)
        velocity = (
            self.w * self.velocity
            + self.phi_p * rp * (self.local_best_location - self.population)
            + self.phi_g * rg * (self.global_best_location - self.population)
        )
        
        # Udpdate the population
        population = self.population + velocity
        self.population = clamp(population, self.lb, self.ub)
        self.velocity = clamp(velocity, self.lb, self.ub)
        
    def _min_by(self, values: List[torch.Tensor],keys: List[torch.Tensor],):
        # Find the value with the minimum key
        values = torch.cat(values, dim=0)
        keys = torch.cat(keys, dim=0)
        min_index = torch.argmin(keys)
        return values[min_index], keys[min_index]
```

### Problem example: Sphere problem

The Sphere problem is a simple, yet fundamental benchmark optimization problem used to test optimization algorithms.

The Sphere function is defined as:

$$
\textbf{min} f(x)= \sum_{i=1}^{n} x_{i}^{2}
$$
**Here is an implementation example of Sphere problem in EvoX:**

```python
import torch

from evox.core import Problem

class Sphere(Problem):
    def __init__(self):
        super().__init__()

    def evaluate(self, pop: torch.Tensor):
        return (pop**2).sum(-1)
```

Now, you can initiate a workflow and run it.

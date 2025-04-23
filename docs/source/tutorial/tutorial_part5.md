# 5. Development and Extension

EvoX not only offers out-of-the-box functionality but also provides developers and advanced users with a rich set of interfaces for custom development and extended integration. This chapter details how to implement custom algorithms and problems, how to utilize EvoX's APIs for deeper control, and how to integrate EvoX with other tools to build more complex applications.

## 5.1 Developing Custom Modules

Sometimes the problem you’re solving or the algorithm you want to use isn’t included in EvoX’s standard library. In such cases, you can develop custom modules using the interfaces EvoX provides.

### 5.1.1 Custom Problems (MyProblem)

If your objective function isn’t available in `evox.problems`, you can define your own by inheriting from the `evox.core.Problem` base class (or conforming to the required interface). A typical problem class needs to implement an `evaluate` function, which receives a batch of solutions (`pop`) and returns the corresponding fitness/objective values. To leverage parallel computation, EvoX requires that `evaluate` support **batch input**.

```python
import torch
from abc import ABC
from typing import Any, Dict
from evox.core.module import ModuleBase

class Problem(ModuleBase, ABC):
    def __init__(self):
        super().__init__()

    def evaluate(self, pop: torch.Tensor) -> torch.Tensor:
        return torch.empty(0)
```

For example, to minimize the sum of cubes of the decision vector:

$$
\min f(x) = \sum_{i=1}^{n} x_i^3
$$

You can implement a `MyProblem` class like this:

```python
import torch
from evox.core import Problem

class MyProblem(Problem):
    def __init__(self):
        super().__init__()

    def evaluate(self, pop: torch.Tensor):
        fitness = torch.sum(pop**3, dim=1)
        return fitness
```

Here, `pop` is a tensor of shape `(population_size, dim)`. The `evaluate` function returns a 1D tensor of fitness values. For multi-objective problems, you can return a dictionary with separate keys for each objective.

You can use your custom problem like a built-in one:

```python
import torch
from MyProblems import MyProblem

popsize = 10
dim = 2
initial_pop = torch.rand(popsize, dim)
problem = MyProblem()
initial_fitness = problem.evaluate(initial_pop)
```

### 5.1.2 Custom Algorithms (MyAlgorithm)

Creating a custom algorithm is more involved, as it includes initialization, generating new solutions, and selection. To create a new algorithm, inherit from `evox.core.Algorithm` and implement at least:

- `__init__`: For initialization.
- `step`: The main evolutionary step logic.

Below is an example of implementing the Particle Swarm Optimization (PSO) algorithm in EvoX:

```python
import torch
from evox.core import Algorithm, Mutable, Parameter
from evox.utils import clamp
from evox.algorithms.so.pso_variants.utils import min_by

class PSO(Algorithm):
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
        device = torch.get_default_device() if device is None else device
        assert lb.shape == ub.shape and lb.ndim == 1

        self.pop_size = pop_size
        self.dim = lb.shape[0]

        lb = lb[None, :].to(device)
        ub = ub[None, :].to(device)
        length = ub - lb

        pop = length * torch.rand(self.pop_size, self.dim, device=device) + lb
        velocity = 2 * length * torch.rand(self.pop_size, self.dim, device=device) - length

        self.lb = lb
        self.ub = ub

        self.w = Parameter(w, device=device)
        self.phi_p = Parameter(phi_p, device=device)
        self.phi_g = Parameter(phi_g, device=device)

        self.pop = Mutable(pop)
        self.velocity = Mutable(velocity)
        self.fit = Mutable(torch.full((self.pop_size,), torch.inf, device=device))
        self.local_best_location = Mutable(pop.clone())
        self.local_best_fit = Mutable(torch.full((self.pop_size,), torch.inf, device=device))
        self.global_best_location = Mutable(pop[0])
        self.global_best_fit = Mutable(torch.tensor(torch.inf, device=device))

    def step(self):
        compare = self.local_best_fit > self.fit
        self.local_best_location = torch.where(compare[:, None], self.pop, self.local_best_location)
        self.local_best_fit = torch.where(compare, self.fit, self.local_best_fit)
        self.global_best_location, self.global_best_fit = min_by(
            [self.global_best_location.unsqueeze(0), self.pop],
            [self.global_best_fit.unsqueeze(0), self.fit],
        )
        rg = torch.rand(self.pop_size, self.dim, device=self.fit.device)
        rp = torch.rand(self.pop_size, self.dim, device=self.fit.device)

        velocity = (
            self.w * self.velocity
            + self.phi_p * rp * (self.local_best_location - self.pop)
            + self.phi_g * rg * (self.global_best_location - self.pop)
        )
        pop = self.pop + velocity
        self.pop = clamp(pop, self.lb, self.ub)
        self.velocity = clamp(velocity, self.lb, self.ub)
        self.fit = self.evaluate(self.pop)

    def init_step(self):
        self.fit = self.evaluate(self.pop)
        self.local_best_fit = self.fit
        self.global_best_fit = torch.min(self.fit)
```

To integrate the algorithm into a workflow:

```python
import torch
from MyProblems import MyProblem
from evox.workflows import EvalMonitor, StdWorkflow
from evox.algorithms import PSO

problem = MyProblem()
algorithm = PSO(
    pop_size=100,
    lb=torch.tensor([-10.0]),
    ub=torch.tensor([10.0])
)
monitor = EvalMonitor()
workflow = StdWorkflow(algorithm, problem, monitor)
for i in range(10):
    workflow.step()
```

### 5.1.3 Custom Other Modules

You can also customize `Monitor`, `Operator`, or any module in EvoX. For example, implement a `MyMonitor` to record population diversity or create a `MyOperator` for custom crossover/mutation strategies. Refer to existing base classes and examples to understand which methods to override.

## 5.2 Using the API

EvoX organizes its APIs into modules, making it easy to extend and combine components.

### 5.2.1 Algorithms and Problems

- **Algorithms**: Found in `evox.algorithms.so` (single-objective) and `evox.algorithms.mo` (multi-objective).

```python
from evox.algorithms.so import PSO
from evox.algorithms.mo import RVEA
```

- **Problems**: Found in `evox.problems`, including:
  - `numerical` – classic test functions (e.g., Ackley, Sphere).
  - `neuroevolution` – RL environments like Brax.
  - `hpo_wrapper` – wrap ML training into HPO problems.

Example: Wrapping a PyTorch MLP with a Brax environment:

```python
import torch.nn as nn
from evox.problems.neuroevolution.brax import BraxProblem

class SimpleMLP(nn.Module):
    ...

policy = SimpleMLP().to(device)
problem = BraxProblem(
    policy=policy,
    env_name="swimmer",
    ...
)
```

Example: Wrapping an optimization process for HPO:

```python
from evox.problems.hpo_wrapper import HPOProblemWrapper
...
hpo_problem = HPOProblemWrapper(
    iterations=30,
    num_instances=128,
    workflow=inner_workflow,
    copy_init_state=True
)
```

### 5.2.2 Workflows and Tools

- **Workflows**: `evox.workflows.StdWorkflow` for basic optimization loops.
- **Monitors**: `EvalMonitor` for tracking performance.

Example:

```python
workflow = StdWorkflow(algorithm, problem, monitor)
compiled_step = torch.compile(workflow.step)
for i in range(10):
    compiled_step()
    print("Top fitness:", monitor.topk_fitness)
```

- **Metrics**: `evox.metrics` provides IGD, Hypervolume, etc.

```python
from evox.metrics import igd
igd_value = igd(current_population, true_pareto_front)
```

- **PyTorch Interoperability**: Seamless integration with `torch.nn`, `torch.Tensor`, etc.

## 5.3 Integration with Other Tools

EvoX is designed to integrate easily with external tools.

### 5.3.1 Machine Learning Integration

Use EvoX to tune hyperparameters:

1. Wrap training/validation as a `Problem`.
2. Use an algorithm like CMA-ES.
3. Optimize hyperparameters over multiple runs.
4. Train final model with the best parameters.

### 5.3.2 Reinforcement Learning Integration

Use EvoX to evolve neural network policies:

1. Wrap RL environment using `BraxProblem`.
2. Flatten policy network using `ParamsAndVector`.
3. Optimize using evolutionary algorithms like GA or CMA-ES.
4. Deploy optimized policies directly or fine-tune with RL.

EvoX supports batch environment simulation to fully utilize GPU/CPU power.

---

**In summary**, EvoX provides powerful, modular APIs and a developer-friendly design for implementing custom algorithms, wrapping any optimization problem, and integrating with ML and RL tools. As you deepen your understanding, you can creatively apply these interfaces to build tailored optimization solutions.

# 7. Practical Examples

This chapter presents several complete, practical examples to demonstrate how to apply the knowledge from previous chapters. We'll build an optimization project from scratch and showcase how EvoX can be integrated with other tools. These examples cover a range of problem types to help you apply EvoX in real-world scenarios.

---

## Example 1: Single-Objective Optimization

**Problem**: Optimize the classic Rastrigin function:

```{math}
f(\mathbf{x}) = 10 d + \sum_{i=1}^{d}[x_i^2 - 10 \cos{(2\pi x_i)}],
```

where $\mathbf{x} \in \mathbb{R}^d$ and $d$ is the dimensionality. The global optimum is 0 at the origin. The function is highly multimodal, making it ideal for testing global optimization algorithms. Here's a plot of the Rastrigin function

```{figure} /_static/rastrigin_function.svg
:alt: A plot of the Rastrigin function
:figwidth: 70%
:align: center

Rastrigin function
```

In this example, we will use the Particle Swarm Optimization (PSO) algorithm to optimize the 10-dimensional Rastrigin function.

**Step 1: Setup**

Assuming you've configured your EvoX environment as explained in Chapter 2.

**Step 2: Workflow Setup**

Create a Python script `opt_rastrigin_10.py`:

```python
import torch
from evox.algorithms.so.pso_variants import PSO
from evox.problems.numerical.basic import Rastrigin
from evox.workflows import StdWorkflow, EvalMonitor
```

Define the PSO algorithm:

```python
dim = 10
algo = PSO(
    pop_size=50,
    lb=-32 * torch.ones(dim),
    ub=32 * torch.ones(dim)
)
```

Set up the problem and workflow:

```python
prob = Rastrigin()
monitor = EvalMonitor()
workflow = StdWorkflow(
    algorithm=algo,
    problem=prob,
    monitor=monitor
)
```

**Step 3: Run Optimization**

```python
workflow.init_step()
for iter in range(501):
    workflow.step()
    if iter % 100 == 0:
        current_best_fitness = monitor.get_best_fitness().item()
        print(f"Iter {iter}, Best Fitness: {current_best_fitness}")

print(f"Final Best Solution: {monitor.get_best_solution()}")
```

**Sample Output**:

```
Iter 0, Best Fitness: 1398.625
Iter 100, Best Fitness: 11.608497619628906
Iter 200, Best Fitness: 2.5700759887695312
Iter 300, Best Fitness: 1.9909820556640625
Iter 400, Best Fitness: 1.9899139404296875
Iter 500, Best Fitness: 0.9976348876953125
Final Best Solution: tensor([...])
```

The PSO algorithm finds a near-optimal solution close to the origin, as expected.

---

## Example 2: Multi-Objective Optimization

**Problem**: Minimize two objectives:

```{math}
f_1(x) = x^2, \quad
f_2(x) = (x - 2)^2
```

The Pareto front lies between $x = 0$ (optimal for $f_1$) and $x = 2$ (optimal for $f_2$).

**Step 1: Environment Setup**

Make sure you have EvoX installed with NSGA-II support.

**Step 2: Define the Custom Problem**

EvoX has many built-in multi-objective test problems, but for this example, we will define a custom problem to optimize the two objectives:

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

from evox.algorithms import NSGA2
from evox.workflows import StdWorkflow, EvalMonitor
# Import evox core classes, see Chapter 5 for details
from evox.core import Problem

class TwoObjectiveProblem(Problem):
    def __init__(
        self,
        d: int = 1,
        m: int = 2,
    ):
        super().__init__()
        self.d = d
        self.m = m

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        x = X[:, 0]
        f_1 = x ** 2
        f_2 = (x - 2) ** 2
        return torch.stack([f_1, f_2], dim=1)

    # Optional: Define the Pareto front function
    def pf(self) -> torch.Tensor:
        pass
```

**Step 3: Define Algorithm and Workflow**

```python
from evox.algorithms import NSGA2
from evox.workflows import StdWorkflow, EvalMonitor

prob = TwoObjectiveProblem()
torch.set_default_device("cuda:0")

algo = NSGA2(
    pop_size=50,
    n_objs=2,
    lb=-5 * torch.ones(1),
    ub=5 * torch.ones(1),
    device=torch.device("cuda"),
)

monitor = EvalMonitor()
workflow = StdWorkflow(algo, prob, monitor)
```

**Step 4: Optimization and Visualization**

```python
workflow.init_step()
for i in range(100):
    workflow.step()

data = algo.fit.cpu().numpy()

import numpy as np
import matplotlib.pyplot as plt

x_vals = np.linspace(0, 2, 400)
pf_f1 = x_vals ** 2
pf_f2 = (x_vals - 2) ** 2

plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c='blue', label='Optimized Population', alpha=0.7)
plt.plot(pf_f1, pf_f2, 'r-', linewidth=2, label='Pareto Front')
plt.xlabel("f1")
plt.ylabel("f2")
plt.title("NSGA-II on Bi-objective Problem")
plt.legend()
plt.grid(True)
plt.show()
```

We can visualize the results using Matplotlib. The blue points represent the optimized population, while the red line shows the Pareto front.

```{figure} /_static/example_nsga2_result.svg
:alt: A plot of the NSGA-II population
:figwidth: 70%
:align: center

A plot of the NSGA-II population after optimization
```

In Jupyter Notebook, you can use EvoX's built-in plotting capabilities to visualize the optimization process and monitor how the population evolves over generations.

```python
monitor.plot()
```

---

## Example 3: Hyperparameter Optimization (HPO)

**Problem**: Tune `C` and `max_iter` of a logistic regression classifier on the breast cancer dataset to maximize validation accuracy.

**Step 1: Load Data and Model**

```python
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from evox.core import Problem

X, y = load_breast_cancer(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
```

**Step 2: Define the Problem**

```python
class HyperParamOptProblem(Problem):
    def __init__(self):
        super().__init__()

    def evaluate(self, pop):
        pop = pop.detach().cpu().numpy()
        objs = []
        for C_val, max_iter_val in pop:
            C_val = float(max(1e-3, C_val))
            max_iter_val = int(max(50, max_iter_val))
            model = LogisticRegression(C=C_val, max_iter=max_iter_val, solver='liblinear')
            model.fit(X_train, y_train)
            acc = model.score(X_val, y_val)
            objs.append(1 - acc)  # error rate
        return torch.tensor(objs)
```

**Step 3: Workflow Setup**

```python
from evox.algorithms.so.es_variants import CMAES
from evox.workflows import EvalMonitor, StdWorkflow

prob = HyperParamOptProblem()
init_params = torch.tensor([1.0, 100.0])
print("Initial error rate:", prob.evaluate(init_params.unsqueeze(0)).item())

algo = CMAES(
    mean_init=init_params,
    sigma=1.0,
)

monitor = EvalMonitor()
workflow = StdWorkflow(algo, prob, monitor)
```

**Step 4: Optimization**

```python
workflow.init_step()
for _ in range(100):
    workflow.step()

best_params = monitor.get_best_solution()
best_error = prob.evaluate(best_params.unsqueeze(0)).item()
print("Optimized error rate:", best_error)
```

**Sample Output**:

```
Initial error rate: 0.0263
Optimized error rate: 0.0088
```

With just a few lines of code, EvoX automates the tedious trial-and-error of hyperparameter tuning.

---

These practical examples illustrate how EvoX can be effectively applied across various domains, from mathematical test functions to machine learning workflows. Once you're comfortable with the basic structure—**Algorithm + Problem + Monitor + Workflow**—you can adapt EvoX to suit almost any optimization task.

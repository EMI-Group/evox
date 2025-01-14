# Transformation from MATLAB to PyTorch and EvoX

This document aims to guide MATLAB users in transitioning to PyTorch and EvoX for evolutionary computation. We will highlight the core differences between MATLAB and PyTorch in terms of syntax, data structures, and workflow. We will then illustrate these differences using a Particle Swarm Optimization (PSO) example in both MATLAB and PyTorch.

## Syntax Differences

### Array Creation and Indexing

#### MATLAB

- Uses 1-based indexing.
- Vectors and matrices are declared using square brackets and semicolons (e.g., `[1 2 3; 4 5 6]`). Random initialization with `rand()` returns values in the interval $[0, 1)$.
- Slicing is performed using the `(start:end)` syntax and utilizes 1-based indexing.

#### PyTorch

- Uses 0-based indexing.
- Arrays (tensors) are typically created using constructors like `torch.rand()`, `torch.zeros()`, or Python lists converted to tensors with `torch.tensor()`.
- Slicing is done using `[start:end]` with 0-based indices.

### Matrix Computation

#### MATLAB

- Performs linear algebraic matrix multiplication by `*`.
- Uses `.*` to multiply corresponding elements of matrices of the same size.
- `/` represents the matrix right division.
- `.^` represents the element-wise power.

#### PyTorch

- Performs linear algebraic matrix multiplication by `@` or `torch.matmul()`.
- Directly uses `*` to multiply corresponding elements of tensors of the same shape or broadcastable shapes.
- `/` represents the element-wise division.
- `**` represents the element-wise power.

### Functions and Definitions

#### MATLAB

- A function is defined by the `function` keyword.
- A file can contain multiple functions, but typically the primary function shares the file name.
- Anonymous functions (e.g., `@(x) sum(x.^2)`) are used for short inline calculations.

#### PyTorch

- Functions are defined using the def keyword, typically within a single `.py` file or module.
- Classes are used to encapsulate data and methods in an object-oriented manner.
- Lambdas serve as short anonymous functions (`lambda x: x.sum()`), but multi-line lambdas are not allowed.

### Control Flow

#### MATLAB

- Uses for `i = 1:N` ... `end` loops with 1-based indexing.
- Conditional statements like `if`, `elseif`, and `else`.

#### PyTorch

- Uses `for i in range(N):` with 0-based indexing.
- Indentation is significant for scoping in loops and conditionals (no `end` keyword).

### Printing and Comments

#### MATLAB

- Uses `fprintf()` functions for formatted output.
- Uses `%` for single-line comments.

#### PyTorch

- Uses `print` with f-strings for formatted output.
- Uses `#` for single-line comments.

## How to Write Evolutionary Computation Algorithm via EvoX?

### MATLAB

A MATLAB code example for PSO algorithm is as follows:
```matlab
function [] = example_pso()
	pso = init_pso(100, [-10, -10], [10, 10], 0.6, 2.5, 0.8);
	test_fn = @(x) (sum(x .* x, 2));
	for i = 1:20
		pso = step_pso(pso, test_fn);
		fprintf("Iteration = %d, global best = %f\n", i, pso.global_best_fitness);
	end
end


function [self] = init_pso(pop_size, lb, ub, w, phi_p, phi_g)
	self = struct();
    self.pop_size = pop_size;
    self.dim = length(lb);
    self.w = w;
    self.phi_p = phi_p;
    self.phi_g = phi_g;
    % setup
    range = ub - lb;
    population = rand(self.pop_size, self.dim);
    population = range .* population + lb;
    velocity = rand(self.pop_size, self.dim);
    velocity = 2 .* range .* velocity - range;
    self.lb = lb;
    self.ub = ub;
    % mutable
    self.population = population;
    self.velocity = velocity;
    self.local_best_location = population;
    self.local_best_fitness = Inf(self.pop_size, 1);
    self.global_best_location = population(1, :);
    self.global_best_fitness = Inf;
end


function [self] = step_pso(self, evaluate)
    % Evaluate
    fitness = evaluate(self.population);
    % Update the local best
    compare = find(self.local_best_fitness > fitness);
    self.local_best_location(compare, :) = self.population(compare, :);
    self.local_best_fitness(compare) = fitness(compare);
    % Update the global best
    values = [self.global_best_location; self.population];
    keys = [self.global_best_fitness; fitness];
    [min_val, min_index] = min(keys);
    self.global_best_location = values(min_index, :);
    self.global_best_fitness = min_val;
    % Update velocity and position
    rg = rand(self.pop_size, self.dim);
    rp = rand(self.pop_size, self.dim);
    velocity = self.w .* self.velocity ...
        + self.phi_p .* rp .* (self.local_best_location - self.population) ...
        + self.phi_g .* rg .* (self.global_best_location - self.population);
    population = self.population + velocity;
    self.population = min(max(population, self.lb), self.ub);
    self.velocity = min(max(velocity, self.lb), self.ub);
end
```
In MATLAB, a primary function `init_pso()` initializes the algorithm, a separate function `step_pso()` performs each iteration step and a main function `example_pso()` orchestrates the loop.

### EvoX
In EvoX, we you construct the PSO algorithm in following way:
```python
import torch

from evox.core import *
from evox.utils import *
from evox.workflows import *
from evox.problems.numerical import Sphere


def main():
    pso = PSO(pop_size=10, lb=torch.tensor([-10.0, -10.0]), ub=torch.tensor([10.0, 10.0]))
    wf = StdWorkflow()
    wf.setup(algorithm=pso, problem=Sphere())
    for i in range(1, 21):
        wf.step()
        print(f"Iteration = {i}, global best = {wf.algorithm.global_best_fitness}")

@jit_class
class PSO(Algorithm):
    def __init__(self, pop_size, lb, ub, w=0.6, phi_p=2.5, phi_g=0.8):
        super().__init__()
        self.pop_size = pop_size
        self.dim = lb.shape[0]
        self.w = w
        self.phi_p = phi_p
        self.phi_g = phi_g
        # setup
        lb = lb[None, :]
        ub = ub[None, :]
        range = ub - lb
        population = torch.rand(self.pop_size, self.dim)
        population = range * population + lb
        velocity = torch.rand(self.pop_size, self.dim)
        velocity = 2 * range * velocity - range
        self.lb = lb
        self.ub = ub
        # mutable
        self.population = population
        self.velocity = velocity
        self.local_best_location = population
        self.local_best_fitness = torch.full((self.pop_size,), fill_value=torch.inf)
        self.global_best_location = population[1, :]
        self.global_best_fitness = torch.tensor(torch.inf)

    def step(self):
        # Evaluate
        fitness = self.evaluate(self.population)
        # Update the local best
        compare = self.local_best_fitness > fitness
        self.local_best_location = torch.where(compare[:, None], self.population, self.local_best_location)
        self.local_best_fitness = torch.where(compare, fitness, self.local_best_fitness)
        # Update the global best
        values = torch.cat([self.global_best_location.unsqueeze(0), self.population], dim=0)
        keys = torch.cat([self.global_best_fitness.unsqueeze(0), fitness], dim=0)
        min_index = torch.argmin(keys)
        self.global_best_location = values[min_index]
        self.global_best_fitness = keys[min_index]
        # Update velocity and position
        rg = torch.rand(self.pop_size, self.dim, dtype=fitness.dtype, device=fitness.device)
        rp = torch.rand(self.pop_size, self.dim, dtype=fitness.dtype, device=fitness.device)
        velocity = (
            self.w * self.velocity
            + self.phi_p * rp * (self.local_best_location - self.population)
            + self.phi_g * rg * (self.global_best_location - self.population)
        )
        population = self.population + velocity
        self.population = clamp(population, self.lb, self.ub)
        self.velocity = clamp(velocity, self.lb, self.ub)


if __name__ == "__main__":
    main()
```
In EvoX, the PSO logic is encapsulated within a class that inherits from `Algorithm`. This object-oriented design simplifies state management and iteration, and introduces following advantages:
- Inherited `evaluate()` method
    You can simply call `self.evaluate(self.population)` to compute fitness values, rather than manually passing your objective function each iteration.
- Built-In Workflow Integration
    When you register your PSO class with a workflow `StdWorkflow`, it handles iterative calls to [`step()`](#StdWorkflow.step) on your behalf.

By extending `Algorithm`, `__init__()` sets up all major PSO components (population, velocity, local/global best, etc.) in a standard Python class constructor.
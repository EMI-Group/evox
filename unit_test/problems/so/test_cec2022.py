import torch

import os
import sys

current_directory = os.getcwd()
if current_directory not in sys.path:
    sys.path.append(current_directory)
    
from src.problems import CEC2022


if __name__ == "__main__":
    problem_number = 1  # For Rosenbrock Function
    dimensionality = 10

# Initialize the problem
    cec2022_problem = CEC2022(problem_number=problem_number, dimension=10)

# Create a population tensor (e.g., 5 individuals)
    population = torch.randn(5, dimensionality, dtype=torch.float64)

# Evaluate the population
    fitness = cec2022_problem.evaluate(population)

    print(fitness)
import torch

import os
import sys

current_directory = os.getcwd()
if current_directory not in sys.path:
    sys.path.append(current_directory)
    
from src.problems import CEC2022


if __name__ == "__main__":
    dimensionality = [2, 10, 20]
    pop_size = 7

for i in range (1, 13):
    problem_number = i
# Initialize the problem
    for dimension in dimensionality:
        if dimension==2 and i in [6, 7, 8]:
            pass
        else:
            cec2022_problem = CEC2022(problem_number=problem_number, dimension=dimension)

# Create a population tensor (e.g., 5 individuals)
            population = torch.randn(pop_size, dimension)

# Evaluate the population
            fitness = cec2022_problem.evaluate(population)
            print(f"The fitness of No.{i} function with {dimension} dimension is {fitness}")
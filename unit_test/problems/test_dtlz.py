import torch
from evox.problems.numerical import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7


if __name__ == "__main__":
    # Problem dimensions and objectives
    d = 12
    m = 3
    ref_num = 1000

    # Create an instance of the DTLZ1 problem
    problem = DTLZ1(d=d, m=m, ref_num=ref_num)

    # Generate a random population (100 individuals, each with d features)
    # population = torch.rand(100, d)
    population = torch.tensor(
        [
            [
                0.1,
                0.5,
                0.2,
                0.1,
                0.5,
                0.2,
                0.1,
                0.5,
                0.2,
                0.1,
                0.5,
                0.2,
            ],
            [
                0.8,
                0.8,
                0.9,
                0.8,
                0.8,
                0.9,
                0.8,
                0.8,
                0.9,
                0.8,
                0.8,
                0.9,
            ],
        ]
    )

    # Evaluate the population
    fitness = problem.evaluate(population)

    print("Fitness of the population:")
    print(fitness)

    # Get the Pareto front for DTLZ1
    pf = problem.pf()
    print("Pareto front:")
    print(pf)

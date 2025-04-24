# 1. Introduction

## What is EvoX?

EvoX is an open-source evolutionary computation library, mainly used to solve various complex optimization problems. Evolutionary computation is a category of algorithms that simulate natural evolution to search for optimal solutions, including Genetic Algorithms (GA), Evolution Strategies (ES), Particle Swarm Optimization (PSO), etc.

Traditional evolutionary frameworks are often limited by computational resources and programming models, making them inefficient for large-scale problems. EvoX overcomes these challenges by combining **GPU acceleration** and **distributed computing**, offering an efficient and scalable solution that allows users to find better solutions faster in complex search spaces.

## Key Features of EvoX

- **Modular Architecture**: EvoX breaks down the optimization process into independent modules: Algorithm, Problem, Monitor, and Workflow. Users don’t need to worry about low-level parallel implementations—EvoX automatically leverages hardware to boost performance.
- **Distributed Execution**: EvoX supports multi-GPU and even multi-node distributed execution. The same code can run on a single machine or scale up to a GPU cluster with little to no additional parallel programming effort. This means your optimization tasks can easily scale from a laptop to a server cluster environment.
- **Functional Programming Interface**: EvoX provides a functional programming interface that aligns closely with mathematical models. Core algorithms are implemented as pure functions without side effects, simplifying parallelization and debugging. Users only need to implement the required functions as defined by the framework, without manually managing complex algorithm states.
- **Visualization and Monitoring**: EvoX includes rich visualization tools and monitoring modules to track the evolutionary process in real time. It uses a dedicated `.exv` data format for efficient streaming and logging of optimization data and provides user-friendly visualization modules to plot convergence curves and more. These tools give users an intuitive understanding of algorithm performance and convergence status.
- **Extensive Algorithm and Problem Libraries**: EvoX includes over 50 single- and multi-objective evolutionary algorithms and over 100 benchmark optimization problems. Whether it’s classical function optimization, complex engineering challenges, or machine learning tasks like hyperparameter optimization (HPO) and neuroevolution, EvoX provides ready-to-use algorithms and problem interfaces out of the box.

## Use Cases

Thanks to the above features, EvoX is especially suitable for the following scenarios:

- **Large-Scale Parameter Optimization**: For high-dimensional problems with large search spaces, EvoX’s GPU-based parallel computing and efficient algorithms can significantly reduce solving time. Examples include optimizing neural network weights or designing complex system parameters—EvoX can accelerate the process.
- **Multi-Objective Optimization**: When you need to optimize multiple (often conflicting) objectives simultaneously—such as balancing cost and performance in engineering design—EvoX includes a variety of multi-objective evolutionary algorithms (like NSGA-II, RVEA, etc.) to search for the Pareto-optimal set.
- **Hyperparameter Optimization (HPO)**: Searching for the best hyperparameter combinations for machine learning models can be time-consuming. EvoX allows the use of evolutionary strategies to efficiently search for hyperparameter configurations, often finding better solutions faster than grid search or random search.
- **Reinforcement Learning and Neuroevolution**: EvoX natively supports reinforcement learning environments (like OpenAI Gym and Google Brax) and deep learning datasets (such as CIFAR-10). This allows users to train control policies or neural network architectures using evolutionary algorithms (i.e., neuroevolution)—for example, using genetic algorithms to optimize RL policy parameters.
- **Academic Research and Engineering Applications**: For researchers in evolutionary algorithms, EvoX offers a highly flexible platform to implement and test new methods. For engineering optimization tasks (like tuning industrial process parameters or adjusting control systems), EvoX provides a high-performance solver that can obtain near-optimal solutions within a reasonable time frame.

In summary, EvoX is suitable for any optimization task that **requires exploring a large solution space quickly**, as long as the task can be massively parallelized on a GPU. Whether you’re an AI researcher or an engineering developer facing complex optimization problems, EvoX is a powerful tool to improve your solving efficiency.

```{tip}
EvoX can run on many GPU devices, including NVIDIA GPUs and AMD GPUs, or even the GPU in your Mac.
```

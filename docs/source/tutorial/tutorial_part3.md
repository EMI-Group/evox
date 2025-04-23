# 3. Basic Operations

In this chapter, we will guide you through running your first EvoX optimization task, including how to **start EvoX** and **initialize the optimization process**, how to **configure an EvoX project** (selecting algorithms and problems and assembling them), and the commonly used **basic commands** (or methods) to control the optimization process. Through a simple example, you will learn the basic usage of EvoX.

## Starting and Initializing

After verifying the installation, you can start writing optimization scripts using EvoX. You can import EvoX in any Python environment (such as terminal, Jupyter Notebook, IDE, etc.).

First, let’s import EvoX and its related modules, and initialize a simple optimization task. For example, we will use the Particle Swarm Optimization (PSO) algorithm to optimize the classic Ackley function. The Ackley function is a common benchmark function with a known global optimum at \((0,0,\dots,0)\), making it suitable for demonstration.
Here is a minimal EvoX example code that demonstrates how to start and run the optimization:

```python
import torch
from evox.algorithms import PSO                      # Import PSO algorithm
from evox.problems.numerical import Ackley           # Import Ackley optimization problem
from evox.workflows import StdWorkflow, EvalMonitor  # Import standard workflow and monitor

# 1. Define the optimization algorithm and problem
algorithm = PSO(
    pop_size=50,                    # Population size of 50
    lb=-32 * torch.ones(2),         # Decision variable lower bound: 2D vector, each -32
    ub= 32 * torch.ones(2)          # Decision variable upper bound: 2D vector, each 32
)
problem = Ackley()                  # Optimization problem: Ackley function (default dimension matches the algorithm)

# 2. Assemble the workflow and add a monitor to track results
monitor = EvalMonitor()
workflow = StdWorkflow(algorithm, problem, monitor)

# 3. Initialize the workflow
workflow.init_step()  # Initialize the internal state of the algorithm and problem

# 4. Execute optimization iterations
for i in range(100):
    workflow.step()   # Advance the optimization by one step

# 5. Obtain results (e.g., print the optimal value)
best_fitness = monitor.get_best_fitness() # Get the best fitness value from the monitor
print("Iteration completed, current best fitness value found:", float(best_fitness))
```

The above code includes the following steps:
- First, we set the parameters for the PSO algorithm: population size of 50 and a search space in 2D ranging from [-32, 32].
- Then, we define the Ackley problem (the Ackley function is defined as 2D by default).
- We create a standard workflow `StdWorkflow` that **assembles** the algorithm and problem, and pass in a monitor `EvalMonitor` to record the optimization process data.
- Next, we complete the initialization process using `workflow.init_step()`, which automatically initializes the population, random seed, and other internal states.
- Then, we run a loop to continuously execute 100 iterations using `workflow.step()`. Each time `step()` is called, the algorithm generates new solutions and evaluates their fitness, continuously approaching the optimal solution.
- Finally, we use the method `get_min_fitness()` provided by the monitor to obtain the best fitness value during the iteration process and print it out.

When you run this script, you will see the output of the optimization iterations, for example:

```text
Iteration completed, current best fitness value found: 9.5367431640625e-07
```

Since we did not explicitly print the intermediate results in the loop, the intermediate results will not be displayed. However, you can judge whether the algorithm has converged based on the final fitness value. For example, the optimal value of the Ackley function is 0, and if the output is close to 0, it indicates that PSO has found a solution close to the global optimum. You can also call `print(monitor.history)` to view the historical data recorded by the monitor or use `monitor.plot()` to plot convergence curves (requires visualization support like Plotly).

```{note}
`StdWorkflow` is a **standard optimization process** encapsulation provided by EvoX. It internally implements the "initialization-iteration update" logic found in traditional evolutionary algorithms and encapsulates the interaction between the algorithm and the problem. For most simple applications, using `StdWorkflow` directly will suffice. The `EvalMonitor` is a monitor that implements methods like `get_best_fitness()` and `plot()` to collect and display performance metrics during the optimization process. Beginners can temporarily understand it as a record book that records the best results of each iteration for later analysis.
```

In the example above, we've created a basic configuration for an EvoX project, including selecting an algorithm, defining the problem, and assembling the workflow. Generally, configuring an EvoX project involves the following steps:

1. **Select/Define an Optimization Problem**: Clarify what optimization problem you're trying to solve. For example, if you're optimizing a mathematical function, EvoX provides many **built-in problems** under the `evox.problems` module (e.g., classic functions like Sphere, Rastrigin, Ackley) that you can use directly. If your problem isn't covered by the built-ins, you can define your own (covered in a later chapter). When configuring a problem, you usually need to know the **dimension of the decision variables** and their **value range**.

2. **Select/Configure an Optimization Algorithm**: Choose an appropriate evolutionary algorithm based on the problem type. EvoX provides a rich set of algorithms under `evox.algorithms`, including single-objective algorithms (like PSO, GA, CMA-ES) and multi-objective algorithms (like NSGA-II, RVEA). After choosing the algorithm, you’ll generally need to set algorithm parameters, such as population size (`pop_size`) and algorithm-specific parameters (like crossover probability and mutation probability in GA). Most algorithms require the **variable range** (lower bound `lb` and upper bound `ub`) and problem dimension to initialize the population. If you're using a multi-objective algorithm, you'll also need to specify the number of objectives (`n_objs`). EvoX's algorithms often provide default values for common hyperparameters, but beginners should consider tuning these parameters based on the task for better performance.

3. **Assemble the Workflow**: With the algorithm and problem instance ready, you need to "assemble" them into a workflow, which represents the full optimization process control. In EvoX, `StdWorkflow` is typically used to combine the algorithm and the problem. If you want to monitor optimization progress, you can add a monitor (like `EvalMonitor`) to the workflow. A monitor isn't required, but it can be very helpful during debugging and analysis. Assembling the workflow usually takes one line of code, like: `workflow = StdWorkflow(algo, prob, monitor)`.

4. **Initialize**: Call the workflow’s initialization method to begin optimization. The latest version of EvoX provides a convenient `StdWorkflow.init_step()` method that completes the initialization process in one call.

5. **Run Iterations**: Use a loop to repeatedly call `workflow.step()` to drive the evolutionary process forward. Each call performs one iteration, including steps like “generate new solutions -> evaluate -> select” inside the algorithm. During iterations, you can use a monitor to observe real-time results, like printing the current best fitness every few generations. Termination criteria can be set based on your needs — common ones include a fixed number of generations (e.g., run for 100 generations), or stopping when monitored metrics converge (e.g., no significant improvement over several generations).

6. **Obtain Results**: After iterations end, you need to extract the final results from the algorithm — such as the best solution and its objective value. In EvoX, these are typically obtained via the monitor. For example, `EvalMonitor.get_best_fitness()` returns the best fitness value. To get the best solution vector, one way is to have the problem object store the best candidate during evaluation, or use the monitor’s interface. In EvoX’s standard implementation, `EvalMonitor` records the best individual and fitness each generation, accessible through its properties. Assuming `monitor.history` stores the history, you can retrieve the best individual from the last generation. Of course, you can also skip `EvalMonitor` and directly query the algorithm object after the loop — this depends on the algorithm implementation. If your custom algorithm implements `get_best()` or stores the best individual in its state, you can extract it directly. However, since EvoX emphasizes pure functions and modularity, results are usually accessed via monitoring modules.

By following these steps, you can clearly structure your optimization task code. For beginners, it's important to understand how the **algorithm-problem-workflow** trio works together: the algorithm handles generating and improving solutions, the problem evaluates their quality, and the workflow connects them into an iterative loop.

Next, we’ll introduce some basic commands and function calls available in EvoX to help deepen your understanding of the optimization process.

## Basic Command Overview

When using EvoX, there are some **commonly used methods and functions** that act as “commands” you’ll want to be familiar with:

### Workflow-Related Methods

- **`StdWorkflow.init_step()`**: Initialization. This is a quick-start command for launching the optimization process, often used at the beginning of a script. It calls the initialization logic for both the algorithm and problem, generates the initial population, and evaluates fitness. After this, the workflow contains the initial state and is ready for iteration.

- **`StdWorkflow.step()`**: Advance one step in the optimization. Each call makes the algorithm generate new candidate solutions based on the current population state, evaluate them, and select the next generation. Users typically call this multiple times inside a loop. The `step()` function usually returns nothing (the internal state is updated within the workflow), though older versions may return a new state. For beginners, you can simply call it without worrying about the return value.

### Monitor-Related Methods

Using `EvalMonitor` as an example, the common methods include:

- `EvalMonitor.get_best_fitness()`: Returns the lowest recorded fitness (for minimization problems) or highest fitness (for maximization problems; the monitor usually distinguishes this). Useful for knowing the current best result.
- `EvalMonitor.get_history()` or `monitor.history`: Retrieves the full history, such as the best value from each generation. Useful for analyzing convergence trends.
- `EvalMonitor.plot()`: Plots convergence or performance curves; requires a graphical environment or Notebook. The monitor usually uses Plotly to render graphs, helping you visually assess algorithm performance.
  Internally, the monitor records the number of evaluations and their results each generation — you typically don't need to intervene, just extract the data when needed.

### Algorithm-Related Methods

- `Algorithm.__init__()` method: Initialization method of an algorithm. Variables are usually wrapped using `evox.core.Mutable()` and hyperparameters with `evox.core.Parameter()`.

- `Algorithm.step()` method: In specific scenarios or when using custom algorithms/problems, you might directly call the algorithm’s `step()` method, which typically encapsulates the entire iteration logic of the algorithm.

- `Algorithm.init_step()` method: The `init_step()` method includes the algorithm’s first iteration. If not overridden, it simply calls the `step()` method. For typical cases, the first iteration is no different from others, so many algorithms may not need a custom `init_step()`. But for algorithms involving hyperparameter tuning, you may need to update hyperparameters or related variables here.

### Device and Parallel Control

- `.to(device)` method: If you need to switch computation devices in your program, use PyTorch’s `.to(device)` method to move tensors (`torch.Tensor`) to GPU/CPU (some PyTorch methods like `torch.randn` also need the device specified). Generally, if you set the device using `torch.set_default_device()` to `cuda:0` (assuming your system supports it and EvoX and dependencies are installed correctly — verify with `torch.cuda.is_available()`), most EvoX high-performance parallel computations will run on GPU automatically. When writing custom algorithms, problems, or monitors, if you create new tensors or use device-sensitive PyTorch methods, it’s recommended to explicitly specify the `device` as `cuda:0` or use `torch.get_default_device()` to avoid performance drops from computations spread across different devices.

For beginners, the above methods are sufficient for handling typical optimization tasks. In short: **Initialize problem/algorithm – set up monitor – assemble workflow – run and retrieve results** is the most common EvoX workflow. Mastering these allows you to tackle basic optimization tasks using EvoX.

Before moving on to the next chapter, try modifying the example: switch from PSO to another algorithm, replace the Ackley function with another test function, or use the monitor to extract more information — this will help you appreciate the flexibility of configuring EvoX projects.

# 4. Advanced Features

EvoX offers many **advanced features** to meet more complex needs. After getting familiar with the basics, this chapter introduces how to customize the framework configuration, manage optional plugin modules, and optimize performance—enabling you to extend and tune EvoX when necessary.

## Custom Configuration

EvoX's default settings suit most situations, but sometimes you might want to customize the framework’s behavior or parameters. For example:

- **Tuning Algorithm Parameters**: Beyond basic population size and number of iterations, many algorithms expose advanced parameters. For example, CMA-ES allows configuration of the initial covariance matrix, and NSGA-II exposes crowding distance parameters. You can pass parameters to the algorithm's constructor, e.g., `GA(crossover_prob=0.9, mutation_prob=0.1)` customizes crossover and mutation probabilities in Genetic Algorithm. Tuning these can fine-tune performance. Refer to the EvoX documentation for each algorithm’s API, where available parameters and defaults are listed.

- **Replacing Operator Components**: You can replace internal evolutionary operators (e.g., selection or mutation strategies). Some algorithm classes support passing custom operator objects. For example, Differential Evolution (DE) may support custom mutation operators, allowing you to provide a custom function or `Operator` class. EvoX’s modular design supports this "plugin-style" replacement. This typically requires understanding the algorithm internals and isn't usually necessary for standard use cases.

- **Multi-Objective Settings**: For multi-objective optimization, you might need to configure preferences or weights—for example, setting weight vectors for weighted sum methods or adjusting reference points during evolution. These configurations are typically exposed via parameters in the problem or algorithm class. For instance, `problem = DTLZ2(d=12, m=3)` defines a 12-dimensional, 3-objective problem. Some algorithms allow passing custom reference vectors. Reading algorithm documentation helps you fully leverage such settings.

- **Logging and Output**: The default `EvalMonitor` already logs key optimization metrics. If you need extra info (e.g., population diversity or average fitness per generation), you can customize the monitor or log manually within your loop. For long-running tasks, you may want to log to a file. This can be done using Python’s `logging` library or simple file I/O to append results for later analysis.

In summary, custom configuration means modifying EvoX’s default behavior for a specific task. This usually involves deeper usage of EvoX's API. We’ll cover more in the **development and extension** section. For beginners, just remember: EvoX offers flexible interfaces that let experienced users tweak nearly every detail—but you can also stick with defaults and get results quickly.

## Plugin Management

"Plugins" here refer to optional components or extension modules in EvoX—such as visualization tools, reinforcement learning environment wrappers, and sibling projects in the EvoX ecosystem. Managing plugins in EvoX mainly involves **installing and using optional modules**. Here are some key extensions and how to manage them:

- **Visualization Plugin**: EvoX includes the `evox.vis_tools` module, which contains a `plot` submodule for charting and supports the `.exv` log format for real-time data streams. To use visualization, install EvoX with the `vis` extra: `pip install evox[vis]`. (If not installed initially, you can install later or just run `pip install plotly` to satisfy dependencies.) When using visual tools, you typically call plot functions after the monitor logs data—for example, `EvalMonitor.plot()` uses `vis_tools.plot`. Ensuring this plugin is installed avoids errors from missing libraries like `matplotlib`.

- **Neuroevolution Plugin**: EvoX supports reinforcement learning environments (like the Brax physics engine) and neural individual optimization (neuroevolution). These features are bundled in the `neuroevolution` extension, installed via `pip install "evox[neuroevolution]"`. This includes the Google Brax library, Gym, and more. After installation, you can use wrappers like `BraxProblem` in `evox.problems.neuroevolution` to turn RL environments into optimization problems. Tools like `ParamsAndVector` are also included for flattening PyTorch model parameters into vectors for evolution. Note that Brax only works on Linux or Windows via WSL—native Windows Python may only run on CPU. In short, enabling or disabling EvoX plugins is controlled via installation of specific extras.

- **Sibling Projects**: EvoX has related projects such as EvoRL (focused on evolutionary reinforcement learning) and EvoGP (GPU-accelerated genetic programming). These share EvoX's design philosophy and interface. If your task is RL-heavy, you might prefer these dedicated frameworks. Managing these plugins means ensuring version compatibility and satisfying dependencies. For example, EvoRL uses JAX and Brax, while EvoGP may require symbolic tree libraries. These libraries can usually coexist without conflicts. Think of them as complementary tools that can be called from the main EvoX project—or left out entirely for a lean setup.

- **Custom Plugins**: Thanks to EvoX’s modularity, you can build your own “plugins.” For example, create a custom `Monitor` class to track unique metrics, or a custom `Problem` subclass that wraps a third-party simulator. These effectively extend EvoX’s capabilities. Best practice is to follow EvoX's interface contracts—for instance, ensure your custom `Problem` has an `evaluate()` method, or that your custom `Monitor` inherits from a base class. Once tested, you could even contribute it to EvoX’s future releases.

Overall, plugin management in EvoX is about **flexible extension and dependency control**. As a beginner, during installation, you can decide whether to include the `vis` and `neuroevolution` extensions. If not initially installed, they can be added later. With plugins, you can monitor optimization progress more easily and integrate EvoX with external tools for greater power.

## Performance Optimization

Performance is a major strength of EvoX. Even using the same algorithm, EvoX’s GPU support can boost speed by several orders of magnitude. However, to fully tap into this, you'll want to follow a few tips:

- **Use GPU Parallelism**: First, ensure your code is actually running on the GPU. As noted earlier, install CUDA-enabled PyTorch and move data to GPU devices. If things seem slow, check with `torch.cuda.is_available()`—it should return `True`. If GPU exists but isn't used, it's likely because tensors were created on the CPU by default. Fix this by explicitly setting `device`, or by ensuring input tensors (like `lb`/`ub`) are on CUDA. EvoX will follow the device of these inputs. On multi-GPU systems, EvoX generally uses **one GPU per process**. To leverage multiple GPUs, you can run multiple processes with different GPUs or wait for future support for coordinated multi-GPU execution.

- **Parallel Evaluation**: A key bottleneck in evolutionary algorithms is **fitness evaluation**. Since evaluations are often independent, they can be parallelized. EvoX batches evaluations when possible—for instance, neural network evaluations or polynomial functions can be computed in parallel using the GPU. For custom problems, avoid Python loops—vectorize your evaluation code to process a whole batch of candidates at once. This makes the most of PyTorch’s parallel capabilities. Simply put: make your problem’s `evaluate()` function operate on batches—not individual solutions—for a massive speedup.

- **Compile for Optimization**: PyTorch 2.0 introduced `torch.compile`, which JIT-compiles models/functions for performance gains. If your evaluation logic is complex, consider compiling before running:

  ```python
  jit_state_step = torch.compile(workflow.step())
  ```

  This could significantly improve performance.
  ```{note}
  Compilation adds overhead and isn’t always supported by all functions or problems. Best suited for large-scale, long-running tasks. On Windows, ensure Triton is installed for `torch.compile` to work.
  ```

- **Tune Population Size**: A larger population increases diversity and global search ability—but also increases per-generation computation. Balance quality and speed by tuning `pop_size`. On GPU, you can often increase it without a linear time cost (thanks to parallelism). But too large a size can cause memory issues. If you’re running out of GPU memory, reduce population size or problem dimension, or use FP16 to save space (set via `torch.set_float32_matmul_precision('medium')`).

- **Reduce Python Overhead**: EvoX moves most core computation to `torch.Tensor`, but user-written loops or custom operators should avoid excessive Python-level operations. Avoid frequent prints (high I/O cost), lists, or data type conversions. Keep your code vectorized/tensorized to leverage fast C++/CUDA kernels under the hood and reduce Python interpreter overhead.

- **Distributed Deployment**: For ultra-large problems, consider running across multiple machines. EvoX supports multi-node setups (via backend communication and sharding). While not beginner-friendly, it’s good to know this exists. With a GPU cluster, refer to EvoX’s documentation for distributed deployment. Usually, you’ll need to set environment variables or launch with special scripts. The architecture allows the same code to run on single or multi-node setups. For your first try, simulate it with multiple processes on one machine.

- **Performance Profiling**: To dive deeper, use tools like PyTorch's profiler or Python’s `cProfile` to analyze bottlenecks. This helps you identify whether the time goes into evaluation, selection, or something else—so you can optimize accordingly (e.g., by caching repeated computations). EvoX is built for performance, but real-world tasks may still hit unique bottlenecks that need analysis.

In short, while EvoX is already optimized at the architecture level, users can further boost performance by **using GPUs correctly**, **batch computing**, and **tuning parameters**. While chasing speed, also remember to maintain result quality—balance is key. As you grow more familiar with EvoX, performance tuning will become second nature.

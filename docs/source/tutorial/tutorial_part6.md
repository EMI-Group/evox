# 6. Troubleshooting and Optimization

When using EvoX, you may encounter issues or want to fine-tune your algorithms. This chapter outlines common problems and solutions, along with debugging strategies and performance tuning tips to help you resolve issues and optimize your experience.

---

## 6.1 Common Issues and Solutions

Here are some frequently encountered problems and how to address them:

**(1) Installation or Import Errors**:

- **Symptom**: Error when running `import evox`.
- **Solution**:
  - **Check installation**: Run `pip show evox` to verify. If it’s not installed, check your virtual environment and reinstall.
  - **Missing dependencies**: If you see `ModuleNotFoundError: No module named 'torch'`, install PyTorch as outlined in Chapter 2.
  - **CUDA mismatch**: Ensure your PyTorch version matches your installed CUDA drivers.

**(2) GPU Not Being Used**:

- **Symptom**: EvoX is running on CPU instead of GPU.
- **Solution**:
  - Check with `torch.cuda.is_available()`. If `False`, reinstall a GPU-compatible PyTorch and check CUDA installation.
  - If `True` but EvoX still uses CPU, make sure your tensors are moved to GPU (see Chapter 3 for configuration).

**(3) Out of Memory (RAM/VRAM)**:

- **Symptom**: You see `OutOfMemoryError`.
- **Solution**:
  - Reduce population size, problem dimension, or evaluation frequency.
  - Use float16 (half precision) or batch evaluation splitting.
  - Turn off debug/deterministic modes in PyTorch.
  - Store statistics only instead of full Pareto fronts (for multi-objective).
  - Upgrading hardware is the ultimate fix for memory bottlenecks.

**(4) Convergence Stagnation**:

- **Symptom**: Algorithm gets stuck in a local optimum.
- **Solution**:
  - Increase population diversity (e.g., higher mutation rate).
  - Try different algorithms or parameters.
  - Ensure the objective function is well-defined (not too noisy or flat).
  - Run multiple trials and pick the best—EvoX makes parallel runs easy.

**(5) Poor Optimization Results**:

- **Symptom**: Final results are below expectations.
- **Solution**:
  - **Check problem definition**: Ensure fitness is computed correctly (e.g., signs, scaling).
  - **Algorithm fit**: Try others or tune hyperparameters.
  - **Use convergence curves**:
    - Flatline early → premature convergence.
    - Oscillating → randomness too high.
  - Adjust algorithm settings and analyze behavior over time.

**(6) Backend Conflicts (JAX vs PyTorch)**:

- **Symptom**: Accidentally installed JAX version of EvoX while using PyTorch examples.
- **Solution**: The default `pip install evox` gives you the PyTorch version. If you installed a JAX version, reinstall using PyTorch instructions (see Chapter 2). JAX features are documented separately.

**(7) Version Mismatch**:

- **Symptom**: API calls don't match installed version.
- **Solution**:
  - EvoX updates may change method names (e.g., `ask/tell` → `step`).
  - Use the latest stable version and refer to its documentation.
  - Adjust code to align with your EvoX version or consider upgrading.

---

## 6.2 Debugging Tips

Debugging evolutionary algorithms can be tricky due to their stochastic nature. Here are practical tips:

**(1) Use Small Scale Testing**:

- Reduce population size and iteration count to simplify debugging.
- Example: `pop_size=5`, `iterations=20`.
- Makes it easier to track population behavior and isolate issues.

**(2) Insert Print Statements**:

- Print population fitness, best individuals, and intermediate values.
- For large tensors, print shapes or use `.tolist()` for smaller ones.
- Helps you understand convergence and operator effects.

**(3) Use IDE Breakpoints**:

- Use PyCharm or VS Code to set breakpoints inside algorithm `step()` or evaluation logic.
- Inspect variable values, tensor contents, or state transitions.
- Be cautious with large tensors—limit what you inspect to avoid crashes.

**(4) Unit Test Custom Components**:

- Test crossover/mutation functions separately.
- Use synthetic inputs to validate output shapes and logic before full integration.

**(5) Profile Execution**:

- Use `torch.autograd.profiler.profile` or `time.time()` to measure step timings.
- Helps you locate bottlenecks or infinite loops.
- Identify whether slowdowns are in evaluation or algorithm logic.

**(6) Log Output to File**:

- Write logs to `.csv` files for long runs.
- Include best fitness per generation, diversity stats, etc.
- Useful when crashes prevent console output from being seen.

Overall, debugging EvoX projects requires a balance of correctness checks and result analysis. Focus first on ensuring the algorithm runs properly, then optimize its effectiveness.

---

## 6.3 Performance Tuning Guide

These tips help you squeeze more speed and quality from EvoX:

**(1) Progressive Scaling**:

- **Start small**: Test logic with small inputs.
- **Scale up** gradually and observe how runtime increases.
- **Identify inefficiencies** if scaling is nonlinear (e.g., 10x population → >10x time).

**(2) Monitor Hardware Usage**:

- Use `nvidia-smi` for GPU, `htop` for CPU.
- High GPU utilization (>50%) is ideal.
- Low GPU usage may mean data isn't on GPU or frequent CPU-GPU transfers are slowing things down.

**(3) Adjust Parallelism**:

- Set CPU threads: `torch.set_num_threads(n)`.
- Avoid oversubscription if using multi-threaded evaluation tools.
- For GPU, optimize `DataLoader` threads if using batch environments or datasets.

**(4) Leverage Batch Evaluation**:

- Batch evaluation is faster than per-individual evaluation.
- Always vectorize `Problem.evaluate()` to process entire populations.

**(5) Reduce Python Overhead**:

- Move heavy logic inside `Algorithm` or `Problem`, avoid complex Python code in main loop.
- Use `workflow.step()` for most operations.
- Minimize per-generation diagnostics if they slow down runs.

**(6) Tune Algorithm Choice**:

- Try CMA-ES, GA, PSO, RVEA, etc.—no single algorithm is best for all problems.
- A faster-converging algorithm may save more time than micro-optimizing one that converges slowly.

Performance tuning is iterative. With patience, you can go from hours of runtime to minutes. EvoX gives you plenty of "knobs"—use them wisely to balance speed and solution quality.

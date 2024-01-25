# High VRAM usage

By default, JAX will allocate 75% of the GPU memory regardless of the program you run.
This preallocate is used to avoid memory fragmentation and improve performance.

To disable this behavior, you can use the `XLA_PYTHON_CLIENT_PREALLOCATE=false` environment variable.

For more information, please refer to the [JAX documentation](https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html).

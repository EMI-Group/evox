# Multi-GPU and Distributed Workflow

EvoX has experimental support for distributed workflows, allowing you to run any normal evolutionary algorithms across multiple GPUs or even multiple machines. This can significantly speed up the optimization process, especially for time-consuming problems.

## How to use

To use the distributed workflow, you need to set up a few things:
1. Make sure you have manually fix the seed of the random number generator.
```python
torch.manual_seed(seed)
# Optional: set the seed for numpy
np.random.seed(seed)
# Optional: use deterministic algorithms
torch.use_deterministic_algorithms(True)
```
```{important}
Make sure to set the seed for all random number generators **before** any torch or numpy operations. This ensures that the random number generator is in a known state before any operations are performed.
```
2. Use the `torch.distributed` or `torchrun` command to launch your script. For example:
```bash
torchrun
    --standalone
    --nnodes=1
    --nproc-per-node=$NUM_GPUS
    your_program.py (--arg1 ... train script args...)
```
```{tip}
`torchrun` is the recommended way to launch distributed torch programs. For more information, see the [PyTorch documentation](https://pytorch.org/docs/stable/elastic/run.html).
```

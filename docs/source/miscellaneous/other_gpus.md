# Use Non-NVIDIA GPUs

This guide explains how to use AMD GPUs and Apple Silicon GPUs with PyTorch in the context of EvoX.

While NVIDIA GPUs are a reliable choice and generally offer strong performance, newer models are optimized for deep learning workloads and large language models. Many of their advanced features, such as support for low-precision data types, are currently underutilized in EvoX. In some cases, non-NVIDIA GPUs can provide better performance and lower cost for evolutionary tasks.

## AMD GPU Support

AMD GPU support in PyTorch is provided via ROCm. AMD devices are recognized as `cuda` devices (just like NVIDIA GPUs). To use an AMD GPU:

1. Install the ROCm-compatible version of PyTorch.
2. Use the standard device setup, e.g., `device = torch.device("cuda")`.

No additional changes are needed beyond using the ROCm build.

## Apple Silicon GPU Support

If you own an Apple Silicon Mac, you can leverage the built-in GPU to accelerate your EvoX workloads.
Apple Silicon GPUs are supported via the Metal Performance Shaders (MPS) backend and are accessible using the `mps` device in PyTorch.

To use an Apple Silicon GPU:

1. Ensure you have the MPS-compatible version of PyTorch installed.
2. Move your tensors and models to the `mps` device, e.g., `device = torch.device("mps")`.

```{note}
The `mps` device does **not** support compilation (e.g., `#evox.compile`).
```

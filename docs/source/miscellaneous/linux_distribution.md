# Linux Distribution and GPU Driver

## Choosing a Linux Distribution

Many people assume that an "old and stable" Linux distribution is the best choice for a server. However, this isn't always true—especially for GPU servers.

The stability of a GPU server often depends on the kernel version and the GPU driver. Because GPU hardware evolves rapidly, newer kernels and drivers tend to be more refined, stable, and compatible with recent GPUs. They usually include more bug fixes and better support for the latest hardware. Additional, the jit compilation and optimizations in the latest kernels and drivers are significantly better than in older versions.

For example, while Ubuntu 20.04 is considered a "stable" release, it's now quite dated for GPU workloads. Even the NVIDIA RTX 3090, which isn’t a particularly new GPU, was released in 2020. This means the default drivers provided by Ubuntu 20.04 may not fully support the 3090, potentially leading to compatibility issues.

In most cases, choosing a newer Linux distribution (such as Ubuntu 25.04 offers better support than 22.04).

Another important factor to consider is how well a Linux distribution supports non-open-source (proprietary) software. Some distributions, such as Fedora, prioritize open-source software and may not include proprietary drivers by default—for example, NVIDIA drivers. This can require additional steps to install and configure GPU drivers. Other distributions, like Arch Linux, Debian, Ubuntu, and NixOS, tend to be more flexible and make it easier to install proprietary drivers when needed.

## Installing the GPU Driver

It is generally recommended to install the GPU driver provided by your Linux distribution. These drivers are typically well-tested and integrated with the kernel.

```{warning}
Unless you are highly experienced with GPU drivers and the Linux kernel, you should avoid installing drivers directly from the NVIDIA website, as they may lead to compatibility issues or require additional configuration.
```

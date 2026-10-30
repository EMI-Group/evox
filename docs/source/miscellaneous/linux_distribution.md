# Linux Distribution and GPU Driver

```{warning} Experimental EvoX 2.0 pre-release
These pages target the **EvoX 2.0 pre-release** (import name `evox`, compiled by `evoxc`).
They do **not** cover the EvoX 1.x torch API (`torch.*`), which was removed in 2.0 and is
documented nowhere in these pages.
```

The advice below is about the *host*, not about EvoX: which Linux to install on a GPU machine, and where
the GPU driver should come from. EvoX 2.0 itself needs none of it to run — the default NumPy backend is
pure Python and works on any distribution — but every compiled backend that reaches a GPU goes through
the vendor's driver, so a host with an outdated kernel or driver will show up as a limit on what those
backends can do.

## Choosing a Linux distribution

Many people assume that an "old and stable" Linux distribution is the best choice for a server. For a GPU
server that is not true, because the stability of such a machine depends on the kernel version and the
GPU driver rather than on the distribution's age.

GPU hardware evolves quickly, and newer kernels and drivers tend to be more refined, more stable and
better matched to recent cards: more bug fixes, better support for the latest hardware, and — the part
that matters for numerical work — significantly better code-generation and runtime behaviour than older
releases. A distribution that was current two years ago may not fully support a GPU that shipped last
year, and the driver it packages may predate the compiler features the backends rely on.

As a rule, prefer a newer distribution over an older one for a GPU machine. Ubuntu 20.04 is a good
example of a release that is still supported but dated for GPU work: even an RTX 3090, hardly a new card,
can fall outside what its default drivers support well. A current release, such as Ubuntu 26.04 LTS, is
usually the smoother starting point than an older long-term-support one.

The second factor is how a distribution treats proprietary software. Some, such as Fedora, prioritise
open-source software and do not include proprietary drivers — NVIDIA's, for instance — by default, which
turns driver installation into extra work. Others, such as Arch Linux, Debian, Ubuntu and NixOS, are more
flexible and make the proprietary driver easy to add when you need it.

## Installing the GPU driver

Install the driver your distribution provides. Distribution drivers are tested against the kernel the
distribution ships and are updated together with it.

```{warning}
Unless you are highly experienced with GPU drivers and the Linux kernel, avoid installing the driver
straight from the NVIDIA website. Vendor installers often conflict with distribution packages, are not
rebuilt when the kernel is updated, and leave a machine that needs manual repair after the next upgrade.
```

```{tip} EvoX does not need the CUDA toolkit
EvoX 2.0 is not a CUDA program. Nothing in it compiles kernels with `nvcc` at install time, and the
default NumPy backend never touches a GPU at all. The compiled backends reach the card through the
*driver*: IREE's CUDA HAL and the XLA CUDA plugin both load against the installed driver, and on AMD the
ROCm user-space runtime that the distribution's driver packages provide. So install and update the
driver, not a toolkit, and let the compiler's extra (`pip install "evoxc[iree]"`) — or your PJRT plugin
of choice — supply the compiler side; {doc}`../guide/install/evox` and
{doc}`../guide/advanced/backends` have the extras story.
```

Once the driver is installed, confirm that the runtime sees the hardware *before* you write an experiment
script: `nvidia-smi` on NVIDIA, `rocminfo` on AMD, and `evoxc.devices()` inside Python, which reports the
device kinds the process can actually use ({doc}`selecting_gpu` shows the exact call and what its output
means). If those disagree, fix the host — no EvoX argument can compensate for a driver that does not see
the card.

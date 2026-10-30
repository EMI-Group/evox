# Python Installation Guide

This guide is for those who are new to the Python programming language and want to install it on their
system. It covers the interpreter and the isolated environment you need before installing EvoX.

```{tip}
EvoX 2.0 is a pure-Python package: the default install runs on a built-in NumPy backend, so you only
need a working Python interpreter — no compiler toolchain, no CUDA driver, no PyTorch.
EvoX declares `Python >= 3.10`; we recommend the **latest stable version** of Python.
```

```{note}
The `evoxc` compiler that EvoX runs on declares `Python >= 3.11`. Use Python 3.11 or newer if you can,
so that both requirements are satisfied by the same interpreter.
```

## Install Python interpreter

### Windows Version

Go to [Download Python](https://www.python.org/downloads/) and download the latest version of Python.

```{note}
Make sure to check the box that says "Add Python to PATH" during the installation process.
```

### macOS Version

Download the official installer from [Download Python](https://www.python.org/downloads/) and run it,
or install Python with [Homebrew](https://brew.sh/):

```console
$ brew install python@3.12
```

Homebrew installs into its own prefix (`/opt/homebrew` on Apple silicon, `/usr/local` on Intel) instead
of into macOS's system Python, so the interpreter you put on your `PATH` is the one your projects use.

### Linux Version

Different Linux distributions have different ways to install Python.
It depends on the package manager of your distribution.
Here are some examples:

- Debian/Ubuntu: `apt`
- Arch Linux: `pacman`
- Fedora: `dnf`

On Debian and Ubuntu the virtual-environment support is a separate package, so install it before you
create an environment:

```console
$ sudo apt install python3 python3-venv
```

### Install through `uv`

`uv` is an extremely fast Python package and project manager, working on Windows, Linux and macOS.
We recommend using `uv` to install the Python interpreter as well as managing Python environments.
The detailed installation guide can be found in the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/#installation-methods).

::::{tab-set}
:::{tab-item} Windows
Use `irm` to download the script and execute it with `iex`:

```console
$ powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Changing the [execution policy](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_execution_policies?view=powershell-7.4#powershell-execution-policies) allows running a script from the internet.

Request a specific version by including it in the URL:

```console
$ powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/0.6.16/install.ps1 | iex"
```

:::

:::{tab-item} Linux and macOS
Use `curl` to download the script and execute it with `sh`:

```console
$ curl -LsSf https://astral.sh/uv/install.sh | sh
```

If your system doesn't have `curl`, you can use `wget`:

```console
$ wget -qO- https://astral.sh/uv/install.sh | sh
```

Request a specific version by including it in the URL:

```console
$ curl -LsSf https://astral.sh/uv/0.6.16/install.sh | sh
```
:::

::::

Once `uv` is installed, it can also fetch an interpreter for you, which avoids the platform sections
above entirely:

```console
$ uv python install 3.12
```

## Managing Python Environments

### Why an isolated environment

A virtual environment is a self-contained directory holding one Python interpreter plus the packages
installed into it. EvoX pulls in its own dependencies (the `evoxc` compiler and NumPy), and research
code usually pins several versions of the same library across projects. Installing into an isolated
environment keeps those versions from colliding, keeps your system Python untouched, and makes the
installation reproducible: deleting the environment is a complete uninstall. Use one environment per
project.

### Pip and Venv

`pip` is the package manager for Python. `venv` is the built-in tool for creating virtual environments
in Python.

To create a virtual environment, run the following command in your project directory (`python3` is the
usual name of the interpreter on macOS and Linux; on Windows it is `python`):

```console
$ python3 -m venv .venv
```

This creates a `.venv` directory containing a copy of the Python interpreter and the standard library.
To activate it, run:

```console
$ source .venv/bin/activate # Bash
$ source .venv/bin/activate.fish # Fish
$ .venv\Scripts\activate # Windows
```

This changes your shell prompt to indicate that you are now working inside the virtual environment.
To leave it, run:

```console
$ deactivate
```

This returns you to your system's default Python interpreter with all its installed libraries.

While the environment is active, `pip` installs into that environment only:

```console
$ pip install "evox"
$ pip list
$ pip install --upgrade evox
$ pip uninstall evox
```

None of these commands affect the system-wide Python installation.

### uv

`uv` can not only manage Python versions, but also manage Python environments.
To create a new environment, run the following command:

```console
$ uv venv --python <python_version> # e.g. 3.11, 3.12, ...
```

This creates a `.venv` directory containing a copy of the Python interpreter and the standard library.
To activate the environment, use the same commands as above:

```console
$ source .venv/bin/activate # Bash
$ source .venv/bin/activate.fish # Fish
$ .venv\Scripts\activate # Windows
```

After activating the environment, use `uv pip` to install packages into it:

```console
$ uv pip install "evox"
```

Alternatively, skip activation and let `uv` resolve the environment for you with `uv run python ...`,
or point `uv pip` at it explicitly with `--python .venv`.

Continue with {doc}`evox` to install the package itself.

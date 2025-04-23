# Python Installation Guide

This guide is for those who are new to the Python programming language and want to install it on their system.
It will help you set up the Python environment needed to run EvoX.

```{tip}
EvoX is written in Python, so you will need to have Python installed on your system.
EvoX support Python 3.10 and above, and we recommend using the **latest version** of Python.
```

## Install Python interpreter

### Windows Version

Go to [Download Python](https://www.python.org/downloads/) and download the latest version of Python.

```{note}
Make sure to check the box that says "Add Python to PATH" during the installation process.
```

### Linux Version

Different Linux distributions have different ways to install Python.
It depends on the package manager of your distribution.
Here are some examples:
- Debian/Ubuntu: `apt`
- Archlinux: `pacman`
- Fedora: `dnf`

### Install through `uv`

`uv` is an extremely fast Python package and project manager, is working on Windows, Linux and MacOS.
We recommend using `uv` to install Python interpreter as well as managing Python environments.
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

:::{tab-item} Linux and MacOS
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

## Managing Python Environments

### Pip and Venv

`pip` is the package manager for Python. `venv` is the built-in tool for creating virtual environments in Python.
A virtual environment is a self-contained directory that contains a Python installation for a particular version of Python, plus several additional packages.
This is useful for managing dependencies for different projects separately.

To create a virtual environment, run the following command in your terminal:

```console
$ python -m venv <env_path> # usually <env_path> is a `.venv` directory in your project
```
This will create a new directory called `<env_path>` that contains a copy of the Python interpreter and the standard library.
To activate the virtual environment, run the following command:

```console
$ source <env_path>/bin/activate # Bash
$ source <env_path>/bin/activate.fish # Fish
$ <env_path>\Scripts\activate # Windows
```
This will change your shell prompt to indicate that you are now working inside the virtual environment.
To deactivate the virtual environment, run the following command:

```console
$ deactivate
```
This will return you to your system's default Python interpreter with all its installed libraries.

While the virtual environment is activated, you can use `pip` to install packages into the virtual environment.
For example, to install the latest version of `numpy`, run the following command:

```console
$ pip install numpy
```
This will install `numpy` into the virtual environment, and it will not affect the system-wide Python installation.
To install a specific version of `numpy`, run the following command:

```console
$ pip install numpy==1.23.4
```
This will install version `1.23.4` of `numpy` into the virtual environment.
To list all the packages installed in the virtual environment, run the following command:

```console
$ pip list
```
This will show you a list of all the packages installed in the virtual environment, along with their versions.
To uninstall a package, run the following command:

```console
$ pip uninstall numpy
```
This will uninstall `numpy` from the virtual environment.
To upgrade a package, run the following command:

```console
$ pip install --upgrade numpy
```
This will upgrade `numpy` to the latest version in the virtual environment.

### uv

`uv` can not only manage Python versions, but also manage Python environments.
To create a new Python environment, run the following command:

```console
$ uv venv --python <python_version> # e.g. 3.10, 3.11, ...
```
This will create a new directory called `.venv` that contains a copy of the Python interpreter and the standard library.
To activate the virtual environment, run the following command:

```console
$ source <env_path>/bin/activate # Bash
$ source <env_path>/bin/activate.fish # Fish
$ <env_path>\Scripts\activate # Windows
```

After activating the virtual environment, you can use `uv pip` to install packages into the virtual environment.
For example, to install the latest version of `numpy`, run the following command:

```console
$ uv pip install numpy
```

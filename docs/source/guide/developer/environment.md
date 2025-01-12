# Develop environment

## Clone the repository and install it in editable mode (recommended)

```bash
git clone https://github.com/EMI-Group/evox.git
cd evox
pip install -e ".[test]" # install the package in editable mode with test dependencies
```

## Nix

Enable the Nix environment by running the following command:
```bash
nix develop .
```
This will create a shell with all the necessary dependencies and a `.venv` directory with the Python environment.

## Style guide

EvoX's has the following style guide:
1. Make sure to use [ruff](https://docs.astral.sh/ruff/) to lint your code.
2. Make sure there are no trailing whitespaces.

## Pre-commit

We recommend using [pre-commit](https://pre-commit.com/) to enforce the style guide.
After installing pre-commit, run the following command to install the hooks in your local repository:
```bash
pre-commit install
```

## Run Unit Test

1. prepare the test environment by installing the required packages (e.g., `torch`) in your Python environment
2. run unittest:
```shell
# run all tests
python -m unittest
# run tests in [path], e.g. python -m unittest unit_test/core/test_jit_util.py
python -m unittest [path-to-test-file]
# run a specific test method or module, e.g. python -m unittest unit_test.core.test_jit_util.TestJitUtil.test_single_eval
python -m unittest [path-to-method-or-module]
```

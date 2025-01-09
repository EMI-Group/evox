# Develop environment

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

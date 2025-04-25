# Document Writing Guide

This guide outlines best practices for writing and maintaining documentation across the codebase and supplementary files.

---

## In-Code Documentation (Docstrings)

Docstrings are essential for understanding the purpose, usage, and behavior of your code. Please adhere to the following conventions:

### General Rules

- Document **all public classes, methods, and functions** using docstrings.
- Use **Sphinx-style** docstrings.
- **Do not** include parameter types in the docstring—they are expected to be declared in the function signature using type hints.

### Format and Directives

Use the following directives to describe different elements:

- `:param <name>:` — Describe a parameter.
- `:return:` — Describe the return value.
- `:raises <exception>:` — Describe exceptions the function might raise.

#### Example

```python
def add(a: int, b: int) -> int:
    """
    Add two integers.

    :param a: The first integer.
    :param b: The second integer.
    :return: The sum of the two integers.
    :raises ValueError: If either input is not an integer.
    """
    if not isinstance(a, int) or not isinstance(b, int):
        raise ValueError("Inputs must be integers.")
    return a + b
```

---

## External Documentation (`docs/` Directory)

All project-level documentation is located in the `docs/` directory. These documents support both users and developers by providing guides, examples, and references.

### Format

- Use **Markdown (`.md`)** or **Jupyter Notebooks (`.ipynb`)** for documentation.
- Markdown is preferred for narrative content and static documentation.
- Use Jupyter Notebooks for executable, interactive content (e.g., tutorials or demos).

### Jupyter Notebook Guidelines

- Ensure all notebooks are **fully executable**.
- Always **run all cells** and **save the output** before committing.
- Our CI/CD environment does **not support GPU execution**, so notebooks must be pre-executed locally.

### Markdown & Notebook Directives

Use the following patterns for rich formatting:

- `[name](#ref)` — Internal cross-reference, e.g., `[ModuleBase](#evox.core.module.ModuleBase)` or `[ModuleBase](#ModuleBase)`
- `![Alt Text](path)` — Embed images, e.g., `![Module base](/_static/modulebase.png)`

---

## Translation

The documentation supports multilingual content. Follow the steps below to update or generate translations.

### Updating Translations (e.g., for `zh_CN`)

```bash
cd docs
make gettext
sphinx-intl update -p build/gettext -l zh_CN
cd ..
python docs/fix_output.py
```

# Working with Module in EvoX

A **module** is a fundamental concept in programming that refers to a self-contained unit of code designed to perform a specific task or a set of related tasks.

This notebook will introduce the basic module in EvoX: [`ModuleBase`](#evox.core.module.ModuleBase).

## Introduction to Module

In the [tutorial](#/tutorial/index), we have mentioned the basic running process in EvoX:

<center><b>Initiate an algorithm and a problem -- Set an monitor -- Initiate a workflow -- Run the workflow</b></center>

This process requires four basic class in EvoX:

- [`Algorithm`](#evox.core.components.Algorithm)
- [`Problem`](#evox.core.components.Problem)
- [`Monitor`](#evox.core.components.Monitor)
- [`Workflow`](#evox.core.components.Workflow)


It is necessary to provide a unified module for them. In EvoX, the four classes are all inherited from the base module — [`ModuleBase`](#evox.core.module.ModuleBase).

```{image} /_static/modulebase.png
:alt: Module base
:align: center
```

## ModuleBase class

The [`ModuleBase`](#evox.core.module.ModuleBase) class is inherited from [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#).

There are many methods in this class, and some important methods are here:

| Method            | Signature                                                    | Usage                                                        |
| ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `__init__`        | `(self, ...)`                                                | Initialize the module.                                       |
| `load_state_dict` | `(self, state_dict: Mapping[str, torch.Tensor], copy: bool = False, ...)` | Copy parameters and buffers from `state_dict` into this module and its descendants. It overwrites [`torch.nn.Module.load_state_dict`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict). |
| `add_mutable`     | `(self, name: str, value: Union[torch.Tensor \| nn.Module, Sequence[torch.Tensor \| nn.Module], Dict[str, torch.Tensor \| nn.Module]]) -> None` | Define a mutable value in this module that can be accessed via `self.[name]` and modified in-place. |

## Role of Module

In EvoX, the [`ModuleBase`](#evox.core.module.ModuleBase) could help to:

- **Contain mutable values**

​	This module is an object-oriented one that can contain mutable values.

- **Support functional programming**

​	Functional programming model is supported via `self.state_dict()` and `self.load_state_dict(...)`.

- **Standardize the initialization**:

​	Basically, predefined submodule(s) which will be ADDED to this module and accessed later in member method(s) should be treated as "non-static members", while any other member(s) should be treated as "static members".

​	The module initialization for non-static members are recommended to be written in the overwritten method of `setup` (or any other member method) rather than `__init__`.

## Usage of Module

Specifically, there are some rules for using [`ModuleBase`](#evox.core.module.ModuleBase) in EvoX:

### Static methods

Static methods to be JIT shall be defined like:

```Python
# One example of the static method defined in a Module

@jit
def func(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y
```
### Non-static Methods

If a method with python dynamic control flows like `if` were to be used with `vmap`,
please use [`torch.cond`](https://pytorch.org/docs/main/generated/torch.cond.html#torch.cond) to explicitly define the control flow.

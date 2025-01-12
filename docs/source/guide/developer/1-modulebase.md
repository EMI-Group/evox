# Working with Module in EvoX

A **module** is a fundamental concept in programming that refers to a self-contained unit of code designed to perform a specific task or a set of related tasks.

This notebook will introduce the basic module in EvoX: [`ModuleBase`](#evox.core.module.ModuleBase).

## Introduction to Module

In the [Quick Start Documentation](#/guide/user/1-start) of the [User Guide](#/guide/user/index), we have mentioned the basic running process in EvoX:

<center><b>Initiate an algorithm and a problem -- Set an monitor -- Initiate a workflow -- Run the workflow</b></center>

This process requires four basic class in EvoX:

- [`Algorithm`](#evox.core.components.Algorithm)
- [`Problem`](#evox.core.components.Problems)
- [`Monitor`](evox.core.components.Monitor)
- [`Workflow`](#evox.core.components.Workflow)


It is necessary to provide a unified module for them. In EvoX, the four classes are all inherited from the base module — [`ModuleBase`](#evox.core.module.ModuleBase).

<center>
  <img src="../../_static/modulebase.svg">
</center>

## ModuleBase class

The [`ModuleBase`](#evox.core.module.ModuleBase) class is inherited from [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#).

There are many methods in this class, and some important methods are here:

| Method            | Signature                                                    | Usage                                                        |
| ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `__init__`        | `(self, ...)`                                                | Initialize the module.                                       |
| `setup`           | `(self, ...) -> self`                                        | Module initialization lines should be written in the overwritten method of `setup` rather than `__init__`. |
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
# one example of the static method defined in a Module 
@jit
def func(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y
```
### Non-static Methods

If a method with python dynamic control flows like `if` were to be JIT, a separated static method with `jit(..., trace=False)` or `torch.jit.script_if_tracing` shall be used:

```python
# Set an module inherited from the ModuleBase class
class ExampleModule(ModuleBase):
    
    ...
    
    # An example of one method with python dynamic control flows like "if"
    # The method using jit(..., trace=False)
    @partial(jit, trace=False)
    def static_func(x: torch.Tensor, threshold: float) -> torch.Tensor:
        if x.flatten()[0] > threshold:
            return torch.sin(x)
        else:
            return torch.tan(x)
        
    # The method to be JIT   
    @jit
    def jit_func(self, p: torch.Tensor) -> torch.Tensor:
        return ExampleModule.static_func(p, self.threshold)
    
    ...
    
```

### Supporting for JIT and non-JIT functions

[`ModuleBase`](#evox.core.module.ModuleBase) is usually used with `jit_class` to automatically JIT all non-magic member methods:

```python
@jit_class
class ExampleModule(ModuleBase):
    # This function will be automatically JIT
    def func1(self, x: torch.Tensor) -> torch.Tensor:
        pass

    # Use `torch.jit.ignore` to disable JIT and leave this function as Python callback
    @torch.jit.ignore
    def func2(self, x: torch.Tensor) -> torch.Tensor:
        # you can implement pure Python logic here
        pass

    # JIT functions can invoke other JIT functions as well as non-JIT functions
    def func3(self, x: torch.Tensor) -> torch.Tensor:
        y = self.func1(x)
        z = self.func2(x)
        pass
```

### Examples

An example of one module inherited from the [`ModuleBase`](#evox.core.module.ModuleBase) is like:

```python
class ExampleModule(ModuleBase):
        def setup(self, mut: torch.Tensor):
            self.add_mutable("mut", mut)
            # or
            self.mut = Mutable(mut)
            return self

        @partial(jit, trace=False)
        def static_func(x: torch.Tensor, threshold: float) -> torch.Tensor:
            if x.flatten()[0] > threshold:
                return torch.sin(x)
            else:
                return torch.tan(x)
        @jit
        def jit_func(self, p: torch.Tensor) -> torch.Tensor:
            x = ExampleModule.static_func(p, self.threshold)
            ...
```

For more details, please look through [the Module in EvoX](#evox.core.module).

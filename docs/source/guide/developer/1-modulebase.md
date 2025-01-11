# Working with Module in EvoX

A **module** is a fundamental concept in programming that refers to a self-contained unit of code designed to perform a specific task or a set of related tasks.

This notebook will introduce the basic module in Evox: {class}`ModuleBase <evox.core.module.ModuleBase>`.

## Introduction of Module

Review the basic running process in EvoX we have mentioned in the [Quick Start](https://evox.readthedocs.io/en/evoxtorch-main/guide/user/1-start.html) of the [User Guide](https://evox.readthedocs.io/en/evoxtorch-main/guide/user/index.html):

<center><b>Initiate an algorithm and a problem -- Set an monitor -- Initiate a workflow -- Run the workflow</b></center>

This process requires four basic class in EvoX:

- {class}`Algorithm <evox.Algorithm>`
- {class}`Problem <evox.Problem>`
- {class}`Monitor <evox.Monitor>`
- {class}`Workflow <evox.Workflow>`


It is necessary to provide a unified module for them. In EvoX, the four classes are all inherited from {class}`ModuleBase <evox.core.ModuleBase>`.

{class}`ModuleBase <evox.core.ModuleBase>` is the base module for all algorithms and problems, and also for {class}`Monitor <evox.Monitor>`

and {class}`Workflow <evox.Workflow>`.

![ModuBase](modulebase.drawio.svg)

## ModuleBase class

The {class}`ModuleBase <evox.core.module.ModuleBase>` class is inherited from {class}`torch.nn.Module`.

There are many methods in this class, and some important methods are here:

| Method            | Signature                                                    | Usage                                                        |
| ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `__init__`        | `(self, ...)`                                                | Initialize the module.                                       |
| `steup`           | `(self, ...) -> self`                                        | Module initialization lines should be written in the overwritten method of `setup` rather than `__init__`. |
| `load_state_dict` | `(self, state_dict: Mapping[str, torch.Tensor], copy: bool = False, ...)` | Copy parameters and buffers from `state_dict` into this module and its descendants. It overwrites [`torch.nn.Module.load_state_dict`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict). |
| `add_mutable`     | `(self,name: str, value: Union[torch.Tensor | nn.Module, Sequence[torch.Tensor | nn.Module], Dict[str, torch.Tensor | nn.Module],],) -> None` | Define a mutable value in this module that can be accessed via `self.[name]` and modified in-place. |

## Role of Module

In EvoX, the {class}`ModuleBase <evox.core.module.ModuleBase>` could help to:

- **Contain mutable values** 

​	This module is an object-oriented one that can contain mutable values.

- **Support functional programming**

​	Functional programming model is supported via `self.state_dict()` and `self.load_state_dict(...)`.

- **Standardize the initialization**:

​	Basically, predefined submodule(s) which will be ADDED to this module and accessed later in member method(s) should be treated as "non-static members", while any other member(s) should be treated as "static members".	

​	The module initialization for non-static members are recommended to be written in the overwritten method of `setup` (or any other member method) rather than `__init__`.

## Usage of Module

Specifically, there are some rules for using {class}`ModuleBase <evox.core.module.ModuleBase>` in EvoX:

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

### Examples

An example of one module inherited from the {class}`ModuleBase <evox.core.module.ModuleBase>` is like:

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

For more details, please look through [the code of Module in EvoX](https://evox.readthedocs.io/en/evoxtorch-main/apidocs/evox/evox.core.module.html).

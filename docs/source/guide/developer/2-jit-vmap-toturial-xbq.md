## How to use JIT for a subclass of ModuleBase

For better understanding of this part, we need to explain 3 important functions in EvoX: [`jit_class`](#evox.core.module.jit_class), [`vmap`](#evox.core.jit_util.vmap) and [`jit`](l#evox.core.jit_util.jit).

### [`jit_class`](#evox.core.module.jit_class) function

[`jit_class`](#evox.core.module.jit_class) is a helper function used to Just-In-Time (JIT) script of [`torch.jit.script`](https://pytorch.org/docs/stable/generated/torch.jit.script.html) or trace ([`torch.jit.trace_module`](https://pytorch.org/docs/stable/generated/torch.jit.trace_module.html#torch-jit-trace-module)) all member methods of the input class. 

[`jit_class`](#evox.core.module.jit_class) has two parameters:

- `cls`: the original class whose member methods are to be lazy JIT.
- `trace`: whether to trace the module or to script the module. Default to `False`.

```{note}
1. In many cases, it is not necessary to wrap your custom algorithms or problems with [`jit_class`](#evox.core.module.jit_class), the workflow(s) will do the trick for you.
2. With `trace=True`, all the member functions are effectively modified to return `self` additionally since side-effects cannot be traced. If you want to preserve the side effects, please set `trace=False` and use the `use_state` function to wrap the member method to generate pure-functional (the `use_state` function will be explained in the next part).
3. Similarly, all module-wide operations like `self.to(...)` can only returns the unwrapped module, which may not be desired. Since most of them are in-place operations, a simple `module.to(...)` can be used instead of `module = module.to(...)`.
```

### [`vmap`](#evox.core.jit_util.vmap) function

[`vmap`](#evox.core.jit_util.vmap) function vectorized map the given function to its mapped version. Based on [`torch.vmap`](https://pytorch.org/docs/main/generated/torch.vmap.html), we made many improvements, and you can see [`torch.vmap`](https://pytorch.org/docs/main/generated/torch.vmap.html) for more information.

### [`jit`](#evox.core.jit_util.jit) function

[`jit`](#evox.core.jit_util.jit) compile the given `func` via [`torch.jit.trace`](https://pytorch.org/docs/stable/generated/torch.jit.script.html) (`trace=True`) or [`torch.jit.script`](https://pytorch.org/docs/stable/generated/torch.jit.trace.html) (`trace=False`).

  This function wrapper effectively deals with nested JIT and vector map (`vmap`) expressions like `jit(func1)` -> `vmap` -> `jit(func2)`, preventing possible errors.

```{note}
1. With `trace=True`, `torch.jit.trace` cannot use SAME example input arguments for function of DIFFERENT parameters,e.g., you cannot pass `tensor_a, tensor_a` to `torch.jit.trace`d version of `f(x: torch.Tensor, y: torch.Tensor)`.
2. With `trace=False`, `torch.jit.script` cannot contain `vmap` expressions directly, please wrap them with `jit(..., trace=True)` or `torch.jit.trace`.
```

In the [Working with Module in EvoX](#/guide/developer/1-modulebase), we have briefly introduced some rules about the methods inside a subclass of the [`ModuleBase`](#evox.core.module.ModuleBase) . Now that [`jit_class`](#evox.core.module.jit_class), [`vmap`](#evox.core.jit_util.vmap) and [`jit`](#evox.core.jit_util.jit) have been explained,  we will explain more rules and provide some specific hints.

### Definition of static methods inside the subclass

Inside the subclass, static methods to be JIT shall be defined like:

```Python
# Import Pytorch
import torch

# Import the ModuleBase class from EvoX
from evox.core import ModuleBase, jit

# Set an module inherited from the ModuleBase class
class ExampleModule(ModuleBase):
    
    ...
    
    # One example of the static method defined in a Module 
    @jit
    def func(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y
    
    ...
    
```
### Definition of non-static methods inside the subclass

If a method with **Python dynamic control flows** like `if` were to be JIT, a separated static method with `jit(..., trace=False)` or `torch.jit.script_if_tracing` shall be used:

```python
# Import Pytorch
import torch

# Import the ModuleBase class from EvoX
from evox.core import ModuleBase, jit

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

```{note}
Dynamic control flow in Python refers to control structures that change dynamically based on conditions at runtime.
`if...elif...else` Conditional Statements, `for`loop and `while` loop are all dynamic control flows. If you have to use them when defining non-static Methods inside the subclass of [`ModuleBase`](#evox.core.module.ModuleBase), please follow the above rule. 
```

### Invocation of external methods inside the subclass

Inside the subclass, external JIT methods can be invocated by the class methods to be JIT:

```python
# Import the ModuleBase class from EvoX
from evox.core import ModuleBase

# One example of the JIT method defined outside the module 
@jit
def external_func(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y

# Set an module inherited from the ModuleBase class
class ExampleModule(ModuleBase):
    
    ...

    # The internal method using jit(..., trace=False)
    @partial(jit, trace=False)
    def static_func(x: torch.Tensor, threshold: float) -> torch.Tensor:

        
    # The internal static method to be JIT   
    @jit
    def jit_func(self, p: torch.Tensor) -> torch.Tensor:
        return external_func(p, p)
    
    ...
    
```

### Automatically JIT for the subclass used with jit_class

[`ModuleBase`](#evox.core.module.ModuleBase)  and its subclasses are usually used with [`jit_class`](#evox.core.module.jit_class) to automatically JIT all non-magic member methods:

```python
# Import Pytorch
import torch

# Import the ModuleBase class from EvoX
from evox.core import ModuleBase, jit_class

@jit_class
class ExampleModule(ModuleBase):
    
    ...
    
    # This function will be automatically JIT
    def automatically_JIT_func1(self, x: torch.Tensor) -> torch.Tensor:
        pass

    # Use `torch.jit.ignore` to disable JIT and leave this function as Python callback
    @torch.jit.ignore
    def not_automatically_JIT__func2(self, x: torch.Tensor) -> torch.Tensor:
        # you can implement pure Python logic here
        pass

    # JIT functions can invoke other JIT functions as well as non-JIT functions
    def automatically_JIT_func3(self, x: torch.Tensor) -> torch.Tensor:
        y = self.func1(x)
        z = self.func2(x)
        pass
    
    ...
    
```

### Invocation of external vmapped methods inside the subclass

Inside the subclass, external vmapped methods can be invocated by the class methods to be JIT:

```Python
# Import Pytorch
import torch

# Import the ModuleBase class from EvoX
from evox.core import ModuleBase, jit, vmap


# One example of the JIT vmapped method defined outside the module
@jit
def external_vmap_func(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    
    # The method to be vmapped
    def external_func(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    	return x + y
    
    return vmap(external_func, in_dims=1, out_dims=1, trace=False)(x, y)

# Set an module inherited from the ModuleBase class
class ExampleModule(ModuleBase):
    
    ...    
    
    # The internal class method to be JIT   
    @jit
    def jit_func(self, p: torch.Tensor) -> torch.Tensor:
        return external_vmap_func(p, p)
    
    ...
    
```

```{note}
If method A invokes vmapped method B, then A and all methods invoke method A can not be vmapped again.
```

### Internal vmapped methods inside the subclass

Inside the subclass, internal vmapped methods can be  JIT by using the [`trace_impl`](#evox.core.module.trace_impl):

```Python
# Import Pytorch
import torch

# Import the ModuleBase class from EvoX
from evox.core import ModuleBase, jit, vmap, trace_impl

# Set an module inherited from the ModuleBase class
class ExampleModule(ModuleBase):
    
    ...    
    
    # The internal vmapped class method to be JIT   
    @jit
    def jit_vmap_func(self, p: torch.Tensor) -> torch.Tensor:
        
        # The original method
        # We can not vmap it
    	def func(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    		return x+y
        
        
        # The method to be vmapped
        # We need to use trace_impl to rewrite the original method
        @trace_impl(func)
    	def trace_func(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    		pass
        
        return vmap(func, in_dims=1, out_dims=1, trace=False)(p, p)
    
    ...
    
```

```{note}
If a class method use [`trace_impl`](#evox.core.module.trace_impl), it will be only available in the trace mode. More details about `trace_impl` will be shown in the next part.
```
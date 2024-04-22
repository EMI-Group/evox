# Jit-able components

## A common pitfall in jit

In JAX, it's hard to jump out of a jit-compiled function, meaning if you jit-compile one function,
then all other functions used within this function must also be jit-compiled.

For example, the following code will result in a compilation error.

```python
@jax.jit
def foo(x):
    return bar(x)

def bar(x):
    return x[x > 0] # dynamic index, not jit-able
```

Even though `bar` is not marked with `jax.jit`, it is still compiled as `foo` calls `bar`.
And since `bar` uses the dynamic index, which is not compatible with `jax.jit`, an error will occur.

## Solution

1. jit-compile low-level components, and give high-level components more freedom.
2. Use [`host callback`](https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html) to call a function on CPU in a jit context.

In EvoX, we almost guarantee that all low-level components are jit-compiled (all operators), and high-level components (`Workflow`) can have both jit-compiled variants (e.g. {doc}`StdWorkflow </api/workflows/standard>`) and non-jit-compiled variants (e.g. {doc}`StdWorkflow </api/workflows/non_jit>`).

Please be aware that using callbacks to jump out of the jit context is not free. Data needs to be transferred between CPU and GPU, which can be an overhead.

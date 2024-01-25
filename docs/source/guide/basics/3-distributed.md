# Distribute the workflow

EvoX provides two distributed workflow implementation, one is based on Ray, and the other one is based on jax.distribute.

## RayDistributedWorkflow

RayDistributedWorkflow is built upon Ray. It can be used on any ray cluster. The Ray cluster should be setup before running the EvoX program.

### Setup Ray cluster

Please refer to [Ray's official documentation](https://docs.ray.io/en/latest/cluster/getting-started.html) for guide on setting up an Ray cluster.

Here is a simple way to setup the cluster locally.

- On the head node
  ```bash
  ray start --head
  ```
- On worker nodes
  ```bash
  ray start --address="<your head node's ip>:6379"
  ```

If you only have 1 machine, but multiple devices, then there is nothing needs to be done. Ray will setup itself in this case.

### Setup EvoX

To scale the workflow using multiple machines through Ray, use the {class}`RayDistributedWorkflow <evox.workflows.RayDistributedWorkflow>` instead of StdWorkflow.

First, import `workflows` from evox

```python
from evox import workflows
```

then create your algorithm, problem, monitor object as usual.

```python
algorithm = ...
problem = ...
monitor = ...
```

Now use `RayDistributedWorkflow`
```python
workflow = workflows.RayDistributedWorkflow(
    algorithm=algorithm,
    problem=problem,
    monitors=[monitor],
    num_workers=4, # the number of machines
    options={ # the options that passes to ray
        "num_gpus": 1
    }
)
```

The `RayDistributedWorkflow` also uses the `workflow.step` function to execute iterations. However, under the hood, it employs a distinct approach that allows for the utilization of multiple devices across different machines.

```{tip}
It is recommanded that one set the environment variable `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
By default JAX will pre-allocate 80% of the device's memory.
This variable disables the GPU memory preallocation, otherwise running multiple JAX processes may cause OOM.
For more information, please refer to [JAX's documentation](https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html) on this matter.
```

## StdWorkflow

StdWorkflow is short for "Universal Workflow",
which aims to use pure JAX to build a workflow that fits any requirement.
Since `StdWorkflow` is written in pure JAX, it has less overhead and don't need any additional dependencies.

### Setup EvoX

Use `StdWorkflow` to create an workflow,
and use `enable_distributed` and pass in the state to enable this feature.

```python
key = jax.random.PRNGKey(0) # a PRNGKey
workflow = workflows.StdWorkflow(
  algorithm,
  problem
  monitor
)
state = workflow.init(key) # init as usual

# important: enable this feature
state = workflow.enable_distributed(state)
```

Then, at the start of your program, before any JAX function is called, do this:

```python
jax.distributed.initialize(coordinator_address=..., num_process=...,process_id=...)
```

In this system, the `coordinator` serves as the primary or head node. The total number of participating processes is indicated by `num_process`. The process with `process_id=0` acts as the coordinator.

From more information, please refer to [jax.distributed.initialize](https://jax.readthedocs.io/en/latest/_autosummary/jax.distributed.initialize.html) and [Using JAX in multi-host and multi-process environments](https://jax.readthedocs.io/en/latest/multi_process.html).

### Run in a cluster

Unlike Ray, JAX's doesn't have the concept of cluster or scheduler.
Instead, it offers tools for enabling distributed interactions among multiple JAX instances. JAX follows the SPMD (single program multiple data) paradigm. To initiate a distributed program in JAX, you simply need to run the same script on different machines. For instance, if your program is named `main.py`, you should execute the following command on all participating machines with different `process_id` argument in `jax.distributed.initialize`:

```bash
python main.py
```

```{tip}
To have `process_id` in the argument, one can use `argparse` to parse the argument from the commandline.
For example:

```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('process_id', type=int)
args = parser.parse_args()
jax.distributed.initialize(
    coordinator_address=...,
    num_processes=...,
    process_id=args.process_id,
)
```

Then call `python main.py 0` on the first machine, `python main 1` on the second machine and so on.

```

### Run on a single machine

In addition to distributed execution across multiple machines, `StdWorkflow` also supports running on a single machine with multiple GPUs. In this scenario, communication between different devices is facilitated by `nccl`, which is considerably more efficient than cross-machine communication.

The setup process remains unchanged from the previous instructions mentioned above. However, since you are working with only a single machine, the subsequent step for multiple machines is no longer necessary:

```python
jax.distributed.initialize(coordinator_address=..., num_process=...,process_id=...)
```

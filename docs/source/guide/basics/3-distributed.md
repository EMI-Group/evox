# Distribute the workflow

To scale the workflow using multiple machines, use the `DistributedPipeline` instead of StdPipeline.

```python
algorithm = <your algorithm>
problem = <your problem>

from evox.pipelines import DistributedPipeline

pipeline = DistributedPipeline(
    algorithm=algorithm,
    problem=problem,
    pop_size=100, # the actual population size used by the algorithm
    num_workers=4, # the number of machines
    options={ # the options that passes to ray
        "num_gpus": 1
    }
)
```
# Distributed Training

## Parallel Model

All states are replicated across all devices including population. Then, on every device, a sharded candidates are passed to `problem.evaluate()`, and the fitnesses are shared across all device (by `all_gather`). This ensures all devices share the same state data without explicit synchronization. In other word, this parallel model only accelerate the problem's evaluation part, and cannot reduce the memory consumption. We use it as our default distributed strategy, as it offers EC algorithms maximum flexibility.

## Multiple devices on a single node

Example:

```python
import jax
import jax.tree_util as jtu
from evox import algorithms, problems, workflows
from evox.core.distributed import tree_unpmap

cso = algorithms.CSO(
    lb=jnp.full(shape=(2,), fill_value=-32),
    ub=jnp.full(shape=(2,), fill_value=32),
    pop_size=16*4,
)
ackley = problems.numerical.Ackley()
workflow = workflows.StdWorkflow(cso, ackley)

key = random.PRNGKey(42)
with jax.default_device(devices[0]):
    state = workflow.init(key)

state = workflow.enable_multi_devices(state, devices)

for i in range(100):
    train_info, state = workflow.step(state)
    train_info = tree_unpmap(train_info, workflow.pmap_axis_name)
    print(train_info['transformed_fitness'])
```

## Multiple devices on multiple nodes

Example of script `dist_train.py`

```python

import argparse
import jax

parser = argparse.ArgumentParser()
parser.add_argument('--addr', type=str, default='127.0.0.1:37233')
parser.add_argument('-n', type=int, required=True)
parser.add_argument('-i', type=int, required=True)
args = parser.parse_args()

jax.distributed.initialize(coordinator_address=args.addr, num_processes=args.n, process_id=args.i, initialization_timeout=30)

total_devices = jax.devices()
devices = jax.local_devices()

print(f'total_devices: {total_devices}')
print(f'devices: {devices}')

from evox import algorithms, problems, workflows
from evox.core.distributed import tree_unpmap

cso = algorithms.CSO(
    lb=jnp.full(shape=(2,), fill_value=-32),
    ub=jnp.full(shape=(2,), fill_value=32),
    pop_size=16*30,
)
ackley = problems.numerical.Ackley()
workflow = workflows.StdWorkflow(cso, ackley)

key = jax.random.PRNGKey(42)
state = workflow.init(key)
state = workflow.enable_multi_devices(state, devices)

for i in range(10):
    train_info, state = workflow.step(state)
    train_info = tree_unpmap(train_info, workflow.pmap_axis_name)
    print(train_info['transformed_fitness'])

jax.distributed.shutdown()
```

Run script on each node:

```shell
# node1 with ip 10.233.96.181
python dist_train.py --addr 10.233.96.181:35429 -n 2 -i 0

# node2
python dist_train.py --addr 10.233.96.181:35429 -n 2 -i 1
```
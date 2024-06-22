import numpy as np
import jax
import jax.tree_util as jtu

from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jax.sharding import NamedSharding, PositionalSharding
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map

from jax._src.distributed import global_state

from enum import Enum

from typing import Optional

POP_AXIS_NAME = "POP"


class ShardingType(Enum):
    SHARED_FIRST_DIM = 1
    REPLICATED = 2

    def get_sharding(self, devices=None):
        """
            Parameters
            ----------
            replicate : bool
                If True, replicate the data across all devices.
                If False, shard the data across all devices on the first axis.
        """
        if devices is None:
            devices = jax.devices()

        devices = np.asarray(devices)

        mesh = Mesh(devices, axis_names=(POP_AXIS_NAME,))
        if self == ShardingType.SHARED_FIRST_DIM:
            sharding = NamedSharding(mesh, P(POP_AXIS_NAME))
        elif self == ShardingType.REPLICATED:
            sharding = NamedSharding(mesh, P())
            # sharding = PositionalSharding(devices).replicate()
        else:
            raise ValueError(f"Unknown sharding type: {self}")
        
        return sharding

def all_gather(x, axis_name: Optional[str] = None, **kwargs):
    """
        All-gather the data across all devices
    """
    if axis_name is None:
        return x
    else:
        return jax.lax.all_gather(x, axis_name, **kwargs)
    
def tree_all_gather(tree, axis_name: Optional[str] = None, **kwargs):
    return jax.tree_map(lambda x: all_gather(x, axis_name, **kwargs), tree)


def unpmap(x, axis_name: Optional[str] = None):
    """
        Only work for pmap(in_axes=0, out_axes=0)
        Return the first device's elements
    """
    if axis_name is None:
        return x
    else:
        return x[0]

def tree_unpmap(tree, axis_name: Optional[str] = None):
    return jax.tree_map(lambda x: unpmap(x, axis_name), tree)


def is_dist_initialized():
    # Note: global_state is a JAX internal API
    return global_state.coordinator_address is not None

def get_process_id():
    if is_dist_initialized():
        return global_state.process_id
    else:
        raise RuntimeError("Distributed is not initialized.")
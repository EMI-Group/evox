=======================
Distribute the workflow
=======================

To scale the workflow using multiple machines, use the :class:`DistributedPipeline <evox.workflows.DistributedPipeline>` instead of StdPipeline.

.. code-block:: python

    algorithm = <your algorithm>
    problem = <your problem>

    from evox.workflows import DistributedPipeline

    pipeline = DistributedPipeline(
        algorithm=algorithm,
        problem=problem,
        pop_size=100, # the actual population size used by the algorithm
        num_workers=4, # the number of machines
        options={ # the options that passes to ray
            "num_gpus": 1
        }
    )

.. tip::
    It is recommanded that one set the environment variable ``XLA_PYTHON_CLIENT_PREALLOCATE=false``.
    This variable control disables the GPU memory preallocation, otherwise running multiple JAX processes may cause OOM.
    For more information, please refer to `JAX's documentation <https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html>`_ on this matter.

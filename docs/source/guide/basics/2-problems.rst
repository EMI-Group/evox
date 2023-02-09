==================================
Working with extended applications
==================================

Working with extended applications in EvoX is easy.

Neuroevolution
==============

EvoX currently supports training ANN with torchvision's datasets via :class:`TorchvisionDataset <evox.problems.neuroevolution.TorchvisionDataset>`.

.. code-block:: python

    from evox.problems.neuroevolution import TorchvisionDataset

    problem = TorchvisionDataset(
        root="./", # where to download the dataset
        forward_func=<your neural network's forward function>,
        batch_size=512, # the batchsize
        num_passes=2, # splite the batch and calculate using 2 passes
        dataset_name="cifar10" # the name of the dataset
    )

In the above example, we created a problem using ``cifar10`` dataset,
each individual in the population will be evaluated with a batch size of 512, and since ``num_passes`` is 2,
the batch will be calculated in 2 passes and each pass has a mini-batch size of ``512 / 2 = 256``.

RL
===

EvoX currently supports rl environments from Gym.

.. code-block:: python

    from evox.problems.rl import Gym
    problem = Gym(
        env_name="ALE/Pong-v5", # the environment's name
        env_options={"full_action_space": False}, # the options passes to the environment
        policy=<your policy>,
        num_workers=16, # number of processes
        env_per_worker=4, # the number of environments each process holds
        controller_options={ # the options that passes to ray
            "num_cpus": 1,
            "num_gpus": 0,
        },
        worker_options={ # the options that passes to ray
            "num_cpus": 1, "num_gpus": 1 / 16
        },
        batch_policy=False,
    )
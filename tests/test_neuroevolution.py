import math

import jax
import jax.numpy as jnp
from flax import linen as nn

from evox import algorithms, problems, workflows
from evox.monitors import EvalMonitor
from evox.utils import TreeAndVector, rank_based_fitness


class MyNet(nn.Module):
    """Smallest network possible.
    Used to run the test.
    """

    @nn.compact
    def __call__(self, x):
        batch_size = x.shape[0]
        # downsample the image to 7x7 to save some computation
        x = x[:, ::4, ::4, 0] / 255.0
        x = x.reshape(batch_size, -1)
        x = nn.Dense(16)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return jax.nn.softmax(x)


def test_tfds():
    BATCH_SIZE = 8
    key = jax.random.PRNGKey(42)
    model_key, workflow_key = jax.random.split(key)

    model = MyNet()
    params = model.init(model_key, jnp.zeros((BATCH_SIZE, 28, 28, 1)))

    @jax.jit
    def loss_func(weight, data):
        # a very bad loss function
        images, labels = data["image"], data["label"]
        outputs = model.apply(weight, images)
        labels = jax.nn.one_hot(labels, 10)
        return jnp.mean((outputs - labels) ** 2)

    problem = problems.neuroevolution.TensorflowDataset(
        dataset="fashion_mnist", batch_size=BATCH_SIZE, loss_func=loss_func
    )
    adapter = TreeAndVector(params)
    monitor = EvalMonitor()

    center = adapter.to_vector(params)
    # create a workflow
    workflow = workflows.StdWorkflow(
        algorithm=algorithms.PGPE(
            optimizer="adam",
            center_init=center,
            pop_size=64,
            stdev_init=0.1,
        ),
        problem=problem,
        candidate_transforms=[adapter.batched_to_tree],
        fitness_transforms=[rank_based_fitness],
        monitors=[monitor],
    )
    # init the workflow
    state = workflow.init(workflow_key)
    for i in range(3):
        state = workflow.step(state)

    best_fitness = monitor.get_best_fitness().item()
    assert math.isclose(best_fitness, 0.07662, abs_tol=0.01)

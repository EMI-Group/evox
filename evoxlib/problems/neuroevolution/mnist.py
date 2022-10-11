import numpy as np
import jax
import jax.numpy as jnp
from torchvision import datasets
from torch.utils.data import DataLoader
import optax
from functools import partial

import evoxlib as exl
from evoxlib import Problem
from .models import LeNet


def numpy_collate_fn(batch: list):
    data, labels = list(zip(*batch))
    return np.stack(data), np.array(labels)


def new_permutation(key, ds_size, batch_size):
    perms = jax.random.permutation(key, ds_size)
    perms = perms.reshape(-1, batch_size)
    return perms


class MNIST(Problem):
    def __init__(
        self,
        root,
        batch_size,
        model,
        loss_func=optax.softmax_cross_entropy_with_integer_labels,
        num_workers=0,
    ):
        self.batch_size = batch_size
        if isinstance(model, nn.Module):
            self.model = model

        if model == "LeNet":
            self.model = LeNet()

        self.loss_func = loss_func

        mnist_dataset = datasets.MNIST(root, download=True, transform=np.array)
        self.dataloader = DataLoader(
            mnist_dataset,
            batch_size=batch_size,
            collate_fn=numpy_collate_fn,
            num_workers=num_workers,
        )
        self.iter = iter(self.dataloader)
        self.initial_params = self.model.init(
            jax.random.PRNGKey(0), jnp.zeros((batch_size, 28, 28, 1))
        )

    # def setup(self, key):
    #     return self._start_new_epoch(key)

    def evaluate(self, state, X):
        try:
            data, labels = next(self.iter)
        except StopIteration:
            self.iter = iter(self.dataloader)
            data, labels = next(self.iter)

        data, labels = jnp.array(data), jnp.array(labels)
        losses = jax.jit(jax.vmap(partial(self._calculate_loss, data, labels)))(X)
        return state, losses

    @exl.jit_method
    def _calculate_loss(self, data, labels, X):
        # add channel dim
        data = data[:, :, :, jnp.newaxis]
        output = self.model.apply(X, data)
        loss = jnp.mean(self.loss_func(output, labels))
        return loss

    def _start_new_epoch(self, key):
        key, subkey = jax.random.split(key)
        perms = new_permutation(subkey, self.train_ds_size, self.batch_size)
        return {"step": 0, "key": key, "perms": perms}

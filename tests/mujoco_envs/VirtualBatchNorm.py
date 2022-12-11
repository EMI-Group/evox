import sys
sys.path.append('/home/lishuang/evox')

import evox as ex
import jax
import jax.numpy as jnp
from evox import Stateful
import gym

data = jnp.load("./vbn_data/Ant-v4.npy", allow_pickle=True).item()
print(data)

assert False

class VirtualBatchNormalizer():
    def __init__(self, key, env, batch_size=128, pick_pro=0.01):
        self.key = key
        self.env = env

        self.batch_size = batch_size
        self.pick_pro = pick_pro

        self.obtain()

    def nextSeed(self):
        self.key, subkey = jax.random.split(self.key)
        return int(jax.random.randint(subkey, (1, ), 0, 1000000)[0])

    def nextRand(self):
        self.key, subkey = jax.random.split(self.key)
        return jax.random.uniform(subkey, (1, ))[0]

    def obtain(self):
        sum = 0
        sumOfSquares = 0
        count = 0

        observation, info = self.env.reset(seed=self.nextSeed())
        while count < self.batch_size:
            if self.nextRand() < self.pick_pro:
                count += 1
                print(count)
                sum += observation
                sumOfSquares += observation ** 2

            action = self.env.action_space.sample()
            observation, reward, terminated, truncated, info = self.env.step(action)

            if terminated or truncated:
                observation, info = self.env.reset(seed=self.nextSeed())

        self.mean = sum / count
        self.std = jnp.sqrt(jnp.maximum(sumOfSquares / count - self.mean ** 2, 1e-2))

    def normalize(self, x):
        return (x - self.mean) / self.std

env_name = "Ant-v4"
env = gym.make(env_name)
key = jax.random.PRNGKey(1)
vbn = VirtualBatchNormalizer(key, env, batch_size=1024, pick_pro=0.01)

import os
save_dir = './vbn_data/'

data = {"mean": vbn.mean, "std": vbn.std}
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

jnp.save(f"{save_dir}{env_name}", data)
import jax.numpy as jnp
import bokeh
from bokeh.plotting import figure, show
from bokeh.models import Spinner
from evox.core.module import Stateful

import time
import os

def initialize_log_dir(log_dir, env_name):
    if log_dir is None:
        current_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
        if env_name is not None:
            log_dir = os.path.join('logs', env_name, current_time)
        else:
            log_dir = os.path.join('logs', current_time)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def file_print(log_file, s):
    with open(log_file, 'a+') as f:
        print(s, file=f)

class GymMonitor:
    def __init__(self, need_save=True, log_dir=None, save_interval=50, env_name=None):
        self.iteration = 0
        self.need_save = need_save
        self.global_min_fitness = float("inf")
        if need_save:
            self.log_dir = initialize_log_dir(log_dir, env_name)
            print(f"use {self.log_dir} as log directory.")
            self.save_interval = save_interval
            self.log_file = os.path.join(self.log_dir, "log.txt")
        else:
            print(f"do not save models")

        
    def update(self, state, problem, pop, fitness):
        self.iteration += 1
        # if len(fitness[0]) > 1:
        #     fitness = jnp.asarray([jnp.sum(li) for li in fitness])
        avg_fitness = jnp.average(fitness)
        min_index = jnp.argmin(fitness)
        min_fitness = fitness[min_index]
        max_fitness = jnp.max(fitness)
        var_ = jnp.var(fitness)
        best_agent = pop[min_index]
        if min_fitness < self.global_min_fitness:
            self.global_min_fitness = min_fitness
            if self.need_save:
                self.save_agent("best_agent.npy", best_agent)

        if self.need_save:
            file_print(self.log_file, f"it: {self.iteration}, min: {min_fitness}, max: {max_fitness}, mean: {avg_fitness}, var: {var_}, global_min: {self.global_min_fitness}")

            if self.iteration % self.save_interval == 0:
                self.save_agent(f"best_iteration{self.iteration}", best_agent)
            

        print(f"iteration: {self.iteration}, min_fitness {min_fitness}, avg_fitness: {avg_fitness}")


    def save_agent(self, name, agent):
        file_name = os.path.join(self.log_dir, name)
        jnp.save(file_name, agent)
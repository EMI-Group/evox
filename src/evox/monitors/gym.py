import jax.numpy as jnp
from evox.core.module import Stateful
from evox.operators import non_dominated_sort, crowding_distance_sort
import time
import os


def initialize_log_dir(log_dir, env_name, mo=False, monitor_info=None):
    if log_dir is None:
        current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))
        log_dir = os.path.join("logs")
        if env_name is not None:
            log_dir = os.path.join(log_dir, env_name)
        if monitor_info is not None:
            log_dir = os.path.join(log_dir, str(monitor_info))
        if mo:
            log_dir = os.path.join(log_dir, "mo")

        log_dir = os.path.join(log_dir, current_time)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def file_print(log_file, s):
    with open(log_file, "a+") as f:
        print(s, file=f)


class GymMonitor:
    def __init__(
        self,
        need_save=True,
        log_dir=None,
        save_interval=50,
        env_name=None,
        mo_keys=None,
        monitor_info=None,
    ):
        self.iteration = 0
        self.need_save = need_save
        self.mo_keys = mo_keys
        if mo_keys is None:
            self.global_min_fitness = float("inf")
            if need_save:
                self.log_dir = initialize_log_dir(
                    log_dir, env_name, monitor_info=monitor_info
                )
                print(f"use {self.log_dir} as log directory.")
                self.save_interval = save_interval
                self.log_file = os.path.join(self.log_dir, "log.txt")
            else:
                print(f"do not save models")
        else:
            if need_save:
                self.log_dir = initialize_log_dir(log_dir, env_name, mo=True)
                print(f"use {self.log_dir} as log directory.")
                self.save_interval = save_interval
                self.log_file = os.path.join(self.log_dir, "log.txt")
            else:
                print(f"do not save models")

    def update(self, state, problem, pop, fitness):
        self.iteration += 1
        if self.mo_keys is None:
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
                file_print(
                    self.log_file,
                    f"it: {self.iteration}, min: {min_fitness}, max: {max_fitness}, mean: {avg_fitness}, var: {var_}, global_min: {self.global_min_fitness}",
                )

                if self.iteration % self.save_interval == 0:
                    self.save_agent(f"best_iteration{self.iteration}", best_agent)

            print(
                f"iteration: {self.iteration}, min_fitness {min_fitness}, avg_fitness: {avg_fitness}"
            )
        else:
            sum_fitness = jnp.asarray([jnp.sum(f) for f in fitness])
            min_fitness = jnp.min(sum_fitness)
            max_fitness = jnp.max(sum_fitness)
            avg_fitness = jnp.average(sum_fitness)
            var_ = jnp.var(sum_fitness)
            non_dominated_rank = non_dominated_sort(fitness)
            print(fitness)
            print(non_dominated_rank)

            pareto_front = non_dominated_rank == 0
            if self.need_save:
                if self.iteration % self.save_interval == 0:
                    new_folder_path = os.path.join(
                        self.log_dir, f"iteration{self.iteration}"
                    )
                    os.makedirs(new_folder_path, exist_ok=True)
                    for idx, value in enumerate(pareto_front):
                        if value:
                            file_name = os.path.join(new_folder_path, f"agent{idx}.npy")
                            jnp.save(file_name, pop[idx])
                    file_print(
                        os.path.join(new_folder_path, "fitness.txt"),
                        f"fitness: {fitness}, non_dominated_rank: {non_dominated_rank}",
                    )

            print(
                f"iteration: {self.iteration}, min_fitness {min_fitness}, avg_fitness: {avg_fitness}"
            )
            file_print(
                self.log_file,
                f"it: {self.iteration}, min: {min_fitness}, max: {max_fitness}, mean: {avg_fitness}, var: {var_}",
            )

    def save_agent(self, name, agent):
        file_name = os.path.join(self.log_dir, name)
        jnp.save(file_name, agent)

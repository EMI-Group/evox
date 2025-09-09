"""Implement the Particle agent."""

import torch

from evox.core import Agent, Mutable


class Particle(Agent):
    """A particle in a PSO-like algorithm.
    The particle maintains its position, velocity, personal best position, and personal best fitness.
    """

    def __init__(self, init_position, init_velocity):
        super().__init__()
        self.position = Mutable(init_position)
        self.velocity = Mutable(init_velocity)
        self.best_pos = Mutable(init_position)
        self.best_fit = Mutable(torch.tensor(torch.inf))

    def observe_self(self, fit):
        """Update the particle's personal best position and fitness based on the current fitness."""
        cond = fit < self.best_fit
        self.best_pos = torch.where(cond, self.position, self.best_pos)
        self.best_fit = torch.where(cond, fit, self.best_fit)

import jax.numpy as jnp
import bokeh
from bokeh.plotting import figure, show
from bokeh.models import Spinner
from evox.core.module import Stateful


class FitnessMonitor:
    def __init__(self, n_objects=1, keep_global_best=True):
        # single object for now
        # assert n_objects == 1
        self.n_objects = n_objects
        self.history = []
        self.min_fitness = float("inf")
        self.keep_global_best = keep_global_best

    def update(self, fitness):
        if self.n_objects > 1:
            self.history.append(fitness)
        else:
            if self.keep_global_best:
                self.min_fitness = min(self.min_fitness, jnp.min(fitness).item())
            else:
                self.min_fitness = jnp.min(fitness).item()
            self.history.append(self.min_fitness)
        return fitness

    def show(self):
        plot = figure(
            title="Fitness - Iteration",
            x_axis_label="Iteration",
            y_axis_label="Fitness",
        )
        fitness_line = plot.line(
            list(range(len(self.history))), self.history, line_width=2
        )
        spinner = Spinner(
            title="Line Width",
            low=0,
            high=60,
            step=1,
            value=fitness_line.glyph.line_width,
            width=64,
        )
        spinner.js_link("value", fitness_line.glyph, "line_width")
        layout = bokeh.layouts.layout(
            [
                [spinner],
                [plot],
            ]
        )
        show(layout)

    def get_min_fitness(self):
        return self.min_fitness
    
    def get_history(self):
        return self.history
    
    def get_last(self):
        return self.history[-1]

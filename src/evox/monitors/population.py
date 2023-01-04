import bokeh
import chex
import jax.numpy as jnp
import numpy as np
from bokeh.models import ColumnDataSource, Slider, Spinner, CustomJS
from bokeh.plotting import figure, show
from evox.core.module import Stateful


def get_init_range(data):
    """Given a numpy array, return a tuple (float, float) used for ploting"""

    min_val = np.min(data)
    max_val = np.max(data)
    range = max_val - min_val
    padding = range * 0.1
    return (min_val - padding, max_val + padding)


class PopulationMonitor:
    def __init__(self, n):
        # single object for now
        assert n < 3
        self.n = n
        self.history = []
        self.min_fitness = float("inf")

    def update(self, pop):
        chex.assert_shape(pop, (None, self.n))
        # convert to numpy array to save gpu memory
        self.history.append(np.array(pop).T)
        return pop

    def show(self):
        # format self.history into ColumnDataSource acceptable format
        # use ColumnDataSource is more efficient than passing numpy array directly
        column_history = {}
        for i, (x, y) in enumerate(self.history):
            column_history[f"{i}_x"] = x
            column_history[f"{i}_y"] = y

        column_history = ColumnDataSource(column_history)

        # init data
        x = self.history[0][0, :]
        y = self.history[0][1, :]
        source = ColumnDataSource({"x": x, "y": y})

        plot = figure(
            title="Population",
            x_axis_label="x",
            y_axis_label="y",
            x_range=get_init_range(x),
            y_range=get_init_range(y),
        )
        scatter = plot.circle(x="x", y="y", source=source, size=4)
        slider = Slider(
            title="Adjust iteration",
            start=0,
            end=len(self.history) - 1,
            step=1,
            value=0,
        )

        slider.js_on_change(
            "value",
            CustomJS(
                args={"source": source, "history": column_history},
                code="""
                    const iter = cb_obj.value
                    source.data.x = history.data[`${iter}_x`]
                    source.data.y = history.data[`${iter}_y`]
                    source.change.emit();
                """,
            ),
        )

        spinner = Spinner(
            title="Size",
            low=0,
            high=20,
            step=1,
            value=scatter.glyph.size,
            width=64,
        )
        spinner.js_link("value", scatter.glyph, "size")

        layout = bokeh.layouts.layout(
            [
                [spinner, slider],
                [plot],
            ]
        )
        show(layout)

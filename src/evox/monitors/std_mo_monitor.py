import jax.numpy as jnp
import numpy as np
from ..operators.non_dominated_sort import non_dominated_sort
import jax.experimental.host_callback as hcb


class StdMOMonitor:
    """Standard multi-objective monitor
    Used for multi-objective workflow,
    can monitor fitness and record the pareto front.

    Parameters
    ----------
    record_pf
        Whether to record the pareto front during the run.
        Default to False.
        Setting it to True will cause the monitor to
        maintain a pareto front of all the solutions with unlimited size,
        which may hurt performance.
    record_fit_history
        Whether to record the full history of fitness value.
        Default to True. Setting it to False may reduce memory usage.
    """

    def __init__(
        self, record_pf=False, record_fit_history=True, record_pop_history=False
    ):
        self.record_pf = record_pf
        self.record_fit_history = record_fit_history
        self.record_pop_history = record_pop_history
        self.fitness_history = []
        self.population_history = []
        self.current_population = None
        self.pf_solutions = None
        self.pf_fitness = None
        self.opt_direction = 1  # default to min, so no transformation is needed

    def set_opt_direction(self, opt_direction):
        self.opt_direction = opt_direction

    def record_pop(self, pop, tranform=None):
        if self.record_pop_history:
            self.population_history.append(pop)
        self.current_population = pop

    def record_fit(self, fitness, metrics=None, tranform=None):
        if self.record_fit_history:
            self.fitness_history.append(fitness)

        if self.record_pf:
            if self.pf_fitness is None:
                self.pf_fitness = fitness
            else:
                self.pf_fitness = jnp.concatenate([self.pf_fitness, fitness], axis=0)

            if self.current_population is not None:
                if self.pf_solutions is None:
                    self.pf_solutions = self.current_population
                else:
                    self.pf_solutions = jnp.concatenate(
                        [self.pf_solutions, self.current_population], axis=0
                    )

            rank = non_dominated_sort(self.pf_fitness)
            pf = rank == 0
            self.pf_fitness = self.pf_fitness[pf]
            self.pf_solutions = self.pf_solutions[pf]

    def get_last(self):
        return self.opt_direction * self.fitness_history[-1]

    def get_pf_fitness(self):
        return self.opt_direction * self.pf_fitness

    def get_pf_solutions(self):
        return self.pf_solutions

    def get_history(self):
        return [self.opt_direction * fit for fit in self.fitness_history]

    def plot(self, pf=True, **kwargs):
        """A Built-in plot function for visualizing the population of single-objective algorithm.
        Use plotly internally, so you need to install plotly to use this function.

        If the problem is provided, we will plot the fitness landscape of the problem.
        """
        try:
            import plotly
            import plotly.express as px
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("The plot function requires plotly to be installed.")

        all_fitness = jnp.concatenate(self.fitness_history, axis=0)
        x_lb = jnp.min(all_fitness[:, 0])
        x_ub = jnp.max(all_fitness[:, 0])
        x_range = x_ub - x_lb
        x_lb = x_lb - 0.1 * x_range
        x_ub = x_ub + 0.1 * x_range
        y_lb = jnp.min(all_fitness[:, 1])
        y_ub = jnp.max(all_fitness[:, 1])
        y_range = y_ub - y_lb
        y_lb = y_lb - 0.1 * y_range
        y_ub = y_ub + 0.1 * y_range

        frames = []
        steps = []
        pf_fitness = None
        for i, fit in enumerate(self.fitness_history):
            # it will make the animation look nicer
            if pf == True:
                if pf_fitness is None:
                    pf_fitness = fit
                else:
                    pf_fitness = jnp.concatenate([pf_fitness, fit])
                rank = non_dominated_sort(pf_fitness)
                indices = rank == 0
                fit = pf_fitness[indices]
            indices = jnp.lexsort(fit.T)
            fit = fit[indices]
            scatter = go.Scatter(
                x=fit[:, 0],
                y=fit[:, 1],
                mode="markers",
                marker={"color": "#636EFA"},
            )
            frames.append(go.Frame(data=[scatter], name=str(i)))

            step = {
                "label": i,
                "method": "animate",
                "args": [
                    [str(i)],
                    {
                        "frame": {"duration": 200, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 200},
                    },
                ],
            }
            steps.append(step)

        sliders = [
            {
                "currentvalue": {"prefix": "Generation: "},
                "pad": {"t": 50},
                "steps": steps,
            }
        ]
        fig = go.Figure(
            data=frames[0].data,
            layout=go.Layout(
                sliders=sliders,
                xaxis={"range": [x_lb, x_ub]},
                yaxis={"range": [y_lb, y_ub]},
                updatemenus=[
                    {
                        "type": "buttons",
                        "buttons": [
                            {
                                "args": [
                                    None,
                                    {
                                        "frame": {"duration": 200, "redraw": False},
                                        "fromcurrent": True,
                                    },
                                ],
                                "label": "Play",
                                "method": "animate",
                            },
                            {
                                "args": [
                                    [None],
                                    {
                                        "frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate",
                                    },
                                ],
                                "label": "Pause",
                                "method": "animate",
                            },
                        ],
                    },
                ],
                **kwargs,
            ),
            frames=frames,
        )

        return fig

    def flush(self):
        hcb.barrier_wait()

    def close(self):
        self.flush()

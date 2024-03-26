from evox import Monitor
from .eval_monitor import EvalMonitor


class MultitaskEvalMonitor(Monitor):
    def __init__(
        self,
        num_tasks,
        full_fit_history=True,
        full_sol_history=False,
        topk=1,
        calc_pf=False,
    ):
        self.monitors = []
        for i in range(num_tasks):
            self.monitors.append(
                EvalMonitor(full_fit_history, full_sol_history, topk, calc_pf)
            )

    def hooks(self):
        return ["post_ask", "post_eval"]

    def set_opt_direction(self, opt_directions):
        for monitor, d in zip(self.monitors, opt_directions):
            monitor.set_opt_direction(d)

    def post_ask(self, _state, cand_sols):
        for monitor, sol in zip(self.monitors, cand_sols):
            monitor.post_ask(_state, sol)

    def post_eval(self, _state, _cand_sol, _transformed_cand_sol, fitness):
        for monitor, fit in zip(self.monitors, fitness):
            monitor.post_eval(_state, _cand_sol, _transformed_cand_sol, fit)

    def get_best_fitness(self):
        return [monitor.get_best_fitness() for monitor in self.monitors]

    def get_best_solutions(self):
        return [monitor.get_best_solution() for monitor in self.monitors]
    
    def flush(self):
        for monitor in self.monitors:
            monitor.flush()
            
    def close(self):
        for monitor in self.monitors:
            monitor.close()

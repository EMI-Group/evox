class Monitor:
    """Monitor base class.
    Monitors are used to monitor the evolutionary process.
    They contains a set of callbacks,
    which will be called at specific points during the execution of the workflow.
    Monitor itself lives outside the main workflow, so jit is not required.

    To implements a monitor, implement your own callbacks and override the hooks method.
    The hooks method should return a list of strings, which are the names of the callbacks.
    Currently the supported callbacks are:
    pre_step, post_step, pre_ask, post_ask, pre_eval, post_eval, pre_tell, post_tell, post_step.
    """

    def __init__(self):
        pass

    def set_opt_direction(self, opt_direction):
        pass

    def hooks(self):
        raise NotImplementedError

    def pre_step(self, state):
        pass

    def pre_ask(self, state):
        pass

    def post_ask(self, state, cand_sol):
        pass

    def pre_eval(self, state, cand_sol, transformed_cand_sol):
        pass

    def post_eval(self, state, cand_sol, transformed_cand_sol, fitness):
        pass

    def pre_tell(
        self, state, cand_sol, transformed_cand_sol, fitness, transformed_fitness
    ):
        pass

    def post_tell(self, state):
        pass

    def post_step(self, state):
        pass

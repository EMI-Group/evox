from abc import ABC
from module import *
from algorithm import Algorithm
from problem import Problem


class Workflow(ModuleBase, ABC):
    """The base class for workflow."""
    
    def step(self) -> ModuleBase:
        """The basic function to step a workflow.

        Usually consists of sequence invocation of `algorithm.ask()`, `problem.evaluate()`, and `algorithm.tell()`.
        
        Returns:
            (`Workflow`): This workflow
        """
        raise NotImplementedError()
    
    def loop(self, max_iterations: int) -> None:
        """Loop the workflow until the maximum number of iterations (`max_iterations`) is reached.

        Args:
            max_iterations (`int`): The desired maximum number of iterations.
        """
        raise NotImplementedError()


if __name__ == "__main__":
    class BasicProblem(Problem):
        def __init__(self):
            super().__init__(num_objective=1)
            self._eval_fn = vmap(BasicProblem._single_eval, in_dims=(0, None), example_shapes=((13, 3), None))
        
        def _single_eval(x: torch.Tensor, p: float = 2.0):
            return (x ** p).sum()
        
        def evaluate(self, pop):
            return self._eval_fn(pop)

    class BasicAlgorithm(Algorithm):
        def __init__(self, pop_size: int, lb: torch.Tensor, ub: torch.Tensor):
            super().__init__(pop_size=pop_size)
            assert lb.ndim == 1 and ub.ndim == 1, f"Lower and upper bounds shall have ndim of 1, got {lb.ndim} and {ub.ndim}"
            assert lb.shape == ub.shape, f"Lower and upper bounds shall have same shape, got {lb.ndim} and {ub.ndim}"
            self.lb = lb
            self.ub = ub
            self.pop = nn.Buffer(torch.empty(pop_size, lb.shape[0], dtype=lb.dtype, device=lb.device))
            self.fit = nn.Buffer(torch.empty(pop_size, dtype=lb.dtype, device=lb.device))
        
        def ask(self):
            pop = torch.rand(self.pop_size, self.lb.shape[0], dtype=self.lb.dtype, device=self.lb.device)
            pop = pop * (self.ub - self.lb)[torch.newaxis, :] + self.lb[torch.newaxis, :]
            self.pop = pop
            return self.pop
        
        def tell(self, fitness):
            self.fit = fitness

    @jit_class
    class BasicWorkflow(Workflow):
        def __init__(self, algorithm: Algorithm, problem: Problem, device: Optional[Union[str, torch.device, int]] = None):
            super().__init__()
            algorithm = algorithm.to(device=device)
            problem = problem.to(device=device)
            self.algorithm = algorithm
            self.problem = problem
            self.generation = nn.Buffer(torch.zeros((), dtype=torch.int32, device=device))
        
        def step(self):
            if torch.jit.is_tracing():
                population, algo = torch.cond(self.generation > 1, self.algorithm.ask, self.algorithm.init_ask, ())
                fitness, prob = self.problem.evaluate(population)
                algo = torch.cond(self.generation > 1, algo.tell, algo.init_tell, (fitness,))
                self.algorithm = algo
                self.problem = prob
            else:
                population = self.algorithm.ask() if self.generation > 1 else self.algorithm.init_ask()
                fitness = self.problem.evaluate(population)
                self.algorithm.tell(fitness) if self.generation > 1 else self.algorithm.init_tell(fitness)
                self.generation = self.generation + 1
            return self
        
        def loop(self, max_iterations: int):
            for _ in range(max_iterations):
                self.step()

    
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    algo = BasicAlgorithm(100, -10 * torch.ones(5), 10 * torch.ones(5))
    prob = BasicProblem()
    workflow = BasicWorkflow(algo, prob)
    print(workflow.step.inlined_graph)
    workflow.step()
    print(workflow.algorithm.fit)
    workflow.step()
    print(workflow.algorithm.fit)
    
    workflow = BasicWorkflow(algo, prob)
    workflow.loop(100)
    print(workflow.algorithm.fit)
    
    # workflow = torch.jit.script(BasicWorkflow(algo, prob))
    # workflow.loop_until(lambda wf: (wf.algorithm.fit < 1).any())
    # print(workflow.algorithm.fit)

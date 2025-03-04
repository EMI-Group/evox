import torch

from evox.core import Problem


class ShiftAffineNumericalProblem(Problem):
    """A numerical problem with a shift and affine transformations to the input points."""

    def __init__(self, shift: torch.Tensor | None = None, affine: torch.Tensor | None = None):
        """Initialize the ShiftAffineNumericalProblem.

        :param shift: The shift vector. Defaults to None. None represents no shift.
        :param affine: The affine transformation matrix. Defaults to None. None represents no affine transformation.
        """
        super().__init__()
        if affine is not None:
            assert affine.ndim == 2 and affine.shape[0] == affine.shape[1], "affine must be a square matrix"
            self.affine = affine[None, :, :]
        else:
            self.affine = None
        if shift is not None:
            assert shift.ndim == 1, "shift must be a vector"
            if affine is not None:
                assert affine.shape[0] == shift.shape[0], "affine and shift must have the same dimension"
            self.shift = shift[None, :]
        else:
            self.shift = None

    def evaluate(self, pop: torch.Tensor) -> torch.Tensor:
        """Evaluate the given population by shifting and applying an affine transformation to the input points first, and then evaluating the points with the actual function.

        :param pop: The population of points to evaluate.

        :return: The evaluated fitness of the population.
        """
        if self.shift is not None:
            pop = pop + self.shift
        if self.affine is not None:
            pop = torch.matmul(pop[:, None, :], self.affine).squeeze(1)
        return self._true_evaluate(pop)


def ackley_func(a: float, b: float, c: float, x: torch.Tensor) -> torch.Tensor:
    return (
        -a * torch.exp(-b * torch.sqrt(torch.mean(x**2, dim=1))) - torch.exp(torch.mean(torch.cos(c * x), dim=1)) + a + torch.e
    )


class Ackley(ShiftAffineNumericalProblem):
    """The Ackley function whose minimum is x = [0, ..., 0]"""

    def __init__(self, a: float = 20.0, b: float = 0.2, c: float = 2 * torch.pi, **kwargs):
        """Initialize the Ackley function with the given parameters.

        :param a: The parameter $a$ in the equation. Defaults to 20.0.
        :param b: The parameter $b$ in the equation. Defaults to 0.2.
        :param c: The parameter $c$ in the equation. Defaults to 2 * pi.
        :param **kwargs: The keyword arguments (`shift` and `affine`) to pass to the superclass `ShiftAffineNumericalProblem`.
        """
        super().__init__(**kwargs)
        self.a = a
        self.b = b
        self.c = c

    def _true_evaluate(self, x: torch.Tensor) -> torch.Tensor:
        return ackley_func(self.a, self.b, self.c, x)


def griewank_func(x: torch.Tensor) -> torch.Tensor:
    f = 1 / 4000 * torch.sum(x**2, dim=1) - torch.prod(torch.cos(x / torch.sqrt(torch.arange(1, x.size(1) + 1)))) + 1
    return f


class Griewank(ShiftAffineNumericalProblem):
    """The Griewank function whose minimum is x = [0, ..., 0]"""

    def __init__(self, **kwargs):
        """Initialize the Griewank function with the given parameters.

        :param **kwargs: The keyword arguments (`shift` and `affine`) to pass to the superclass `ShiftAffineNumericalProblem`.
        """
        super().__init__(**kwargs)

    def _true_evaluate(self, x: torch.Tensor) -> torch.Tensor:
        return griewank_func(x)


def rastrigin_func(x: torch.Tensor) -> torch.Tensor:
    return 10 * x.size(1) + torch.sum(x**2 - 10 * torch.cos(2 * torch.pi * x), dim=1)


class Rastrigin(ShiftAffineNumericalProblem):
    """The Rastrigin function whose minimum is x = [0, ..., 0]"""

    def __init__(self, **kwargs):
        """Initialize the Griewank function with the given parameters.

        :param **kwargs: The keyword arguments (`shift` and `affine`) to pass to the superclass `ShiftAffineNumericalProblem`.
        """
        super().__init__(**kwargs)

    def _true_evaluate(self, x: torch.Tensor) -> torch.Tensor:
        return rastrigin_func(x)


def rosenbrock_func(x):
    f = torch.sum(100 * ((x[:, 1:]) - x[:, : x.size(1) - 1] ** 2) ** 2 + (x[:, : x.size(1) - 1] - 1) ** 2, dim=1)
    return f


class Rosenbrock(ShiftAffineNumericalProblem):
    """The Rosenbrock function whose minimum is x = [1, ..., 1]"""

    def __init__(self, **kwargs):
        """Initialize the Griewank function with the given parameters.

        :param **kwargs: The keyword arguments (`shift` and `affine`) to pass to the superclass `ShiftAffineNumericalProblem`.
        """
        super().__init__(**kwargs)

    def _true_evaluate(self, x: torch.Tensor) -> torch.Tensor:
        return rosenbrock_func(x)


def schwefel_func(x):
    return 418.9828872724338 * x.size(1) - torch.sum(x * torch.sin(torch.sqrt(torch.abs(x))), dim=1)


class Schwefel(ShiftAffineNumericalProblem):
    """The Schwefel function whose minimum is x = [420.9687, ..., 420.9687]"""

    def __init__(self, **kwargs):
        """Initialize the Griewank function with the given parameters.

        :param **kwargs: The keyword arguments (`shift` and `affine`) to pass to the superclass `ShiftAffineNumericalProblem`.
        """
        super().__init__(**kwargs)

    def _true_evaluate(self, x: torch.Tensor) -> torch.Tensor:
        return schwefel_func(x)


def sphere_func(x):
    return torch.sum(x**2, dim=1)


class Sphere(ShiftAffineNumericalProblem):
    """The sphere function whose minimum is x = [0, ..., 0]"""

    def __init__(self, **kwargs):
        """Initialize the Griewank function with the given parameters.

        :param **kwargs: The keyword arguments (`shift` and `affine`) to pass to the superclass `ShiftAffineNumericalProblem`.
        """
        super().__init__(**kwargs)

    def _true_evaluate(self, x: torch.Tensor) -> torch.Tensor:
        return sphere_func(x)

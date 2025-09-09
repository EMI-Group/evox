import torch

from evox.agents import Particle
from evox.core import Algorithm, Parameter, use_state, vmap


class PSOAgent(Particle):
    """A particle agent for PSO algorithm.
    The particle maintains its position, velocity, personal best position, and personal best fitness.
    """

    def __init__(self, init_position, init_velocity):
        super().__init__(init_position, init_velocity)
        self.global_best_pos = init_position
        self.global_best_fit = torch.tensor(torch.inf)

    def observe_global(self, global_best_pos, global_best_fit):
        """Update the particle's knowledge of the global best position."""
        self.global_best_pos = global_best_pos
        self.global_best_fit = global_best_fit

    def move(self, w, phi_p, phi_g):
        """Update the particle's velocity and position based on inertia, cognitive, and social components."""
        dim = self.position.shape[0]
        rg = torch.rand(dim, device=self.position.device)
        rp = torch.rand(dim, device=self.position.device)
        self.velocity = (
            w * self.velocity
            + phi_p * rp * (self.best_pos - self.position)
            + phi_g * rg * (self.global_best_pos - self.position)
        )
        self.position = self.position + self.velocity

    def clamp(self, lb, ub):
        """Clamp the particle's position and velocity within the given bounds.
        Here we implement the reflective boundary condition (bounce-back).
        """
        self.position = torch.where(self.position < lb, 2 * lb - self.position, self.position)
        self.position = torch.where(self.position > ub, 2 * ub - self.position, self.position)
        self.velocity = torch.where(self.position < lb, -self.velocity, self.velocity)
        self.velocity = torch.where(self.position > ub, -self.velocity, self.velocity)

    def get_position(self):
        return self.position


def vmap_agent_run(agents, func, *args):
    agents = vmap(use_state(func))(agents, *args)
    return agents


def lockstep(agents, func, args, *, in_dims=0, randomness="different"):
    return vmap(use_state(func), in_dims=in_dims, randomness=randomness)(agents, *args)


class PSO(Algorithm):
    """The basic Particle Swarm Optimization (PSO) algorithm.

    ## Class Methods

    * `__init__`: Initializes the PSO algorithm with given parameters (population size, lower and upper bounds, inertia weight, cognitive weight, and social weight).
    * `step`: Performs a single optimization step using Particle Swarm Optimization (PSO), updating local best positions and fitness values, and adjusting velocity and positions based on inertia, cognitive, and social components.

    Note that the `evaluate` method is not defined in this class, it is a proxy function of `Problem.evaluate` set by workflow; therefore, it cannot be used in class methods other than `step`.
    """

    def __init__(
        self,
        pop_size: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        w: float = 0.6,
        phi_p: float = 2.5,
        phi_g: float = 0.8,
        device: torch.device | None = None,
    ):
        """
        Initialize the PSO algorithm with the given parameters.

        :param pop_size: The size of the population.
        :param w: The inertia weight. Defaults to 0.6.
        :param phi_p: The cognitive weight. Defaults to 2.5.
        :param phi_g: The social weight. Defaults to 0.8.
        :param lb: The lower bounds of the particle positions. Must be a 1D tensor.
        :param ub: The upper bounds of the particle positions. Must be a 1D tensor.
        :param device: The device to use for the tensors. Defaults to None.
        """
        super().__init__()
        device = torch.get_default_device() if device is None else device
        assert lb.shape == ub.shape and lb.ndim == 1 and ub.ndim == 1 and lb.dtype == ub.dtype
        self.pop_size = pop_size
        self.dim = lb.shape[0]
        # Here, Parameter is used to indicate that these values are hyper-parameters
        # so that they can be correctly traced and vector-mapped
        self.w = Parameter(w, device=device)
        self.phi_p = Parameter(phi_p, device=device)
        self.phi_g = Parameter(phi_g, device=device)
        # setup
        lb = lb[None, :].to(device=device)
        ub = ub[None, :].to(device=device)
        length = ub - lb
        pop = torch.rand(self.pop_size, self.dim, device=device)
        pop = length * pop + lb
        velocity = torch.rand(self.pop_size, self.dim, device=device)
        velocity = 2 * length * velocity - length
        # write to self
        self.lb = lb
        self.ub = ub
        # Initialize population and velocity
        self.particles = [PSOAgent(p, v) for p, v in zip(pop, velocity)]
        particle_state = torch.func.stack_module_state(self.particles)
        self.particle_state = particle_state[0] | particle_state[1]

    def step(self):
        """
        Perform a normal optimization step using PSO.

        This function evaluates the fitness of the current population, updates the
        local best positions and fitness values, and adjusts the velocity and
        positions of particles based on inertia, cognitive, and social components.
        It ensures that the updated positions and velocities are clamped within the
        specified bounds.

        The local best positions and fitness values are updated if the current
        fitness is better than the recorded local best. The global best position
        and fitness are determined using helper functions.

        The velocity is updated based on the weighted sum of the previous velocity,
        the cognitive component (personal best), and the social component (global
        best). The population positions are then updated using the new velocities.
        """
        all_local_best_fit = self.particle_state["best_fit"]
        all_local_best_pos = self.particle_state["best_pos"]
        global_best_fit = torch.min(all_local_best_fit)
        global_best_pos = all_local_best_pos[torch.argmin(all_local_best_fit)]
        self.particle_state = lockstep(
            self.particle_state,
            self.particles[0].observe_global,
            in_dims=(0, None, None),
            args=(global_best_pos, global_best_fit),
        )
        self.particle_state = lockstep(
            self.particle_state, self.particles[0].move, in_dims=(0, None, None, None), args=(self.w, self.phi_p, self.phi_g)
        )
        self.particle_state = lockstep(
            self.particle_state, self.particles[0].clamp, in_dims=(0, None, None), args=(self.lb, self.ub)
        )
        _, position = lockstep(self.particle_state, self.particles[0].get_position, ())
        fit = self.evaluate(position)
        self.particle_state = lockstep(self.particle_state, self.particles[0].observe_self, in_dims=0, args=(fit,))

    def init_step(self):
        """Perform the first step of the PSO optimization.

        See `step` for more details.
        """
        _, position = lockstep(self.particle_state, self.particles[0].get_position, ())
        fit = self.evaluate(position)
        self.particle_state = lockstep(self.particle_state, self.particles[0].observe_self, in_dims=0, args=(fit,))

    def record_step(self):
        return self.particles

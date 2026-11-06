import torch

from evox.core import Algorithm, Mutable, Parameter


class PlusLambdaES(Algorithm):
    """Elitist (1+λ)-ES on a boxed search space.

    Each step builds a population of shape ``(1 + λ, dim)``: index 0 is the
    current parent (center); indices ``1…λ`` are offspring
    ``center + σ ε`` clipped to ``[lb, ub]``. Minimization is assumed.

    When ``delta_accept > 0``, the center moves to offspring ``i > 0`` only if
    ``f_i < f_0 * (1 - delta_accept)`` (conservative acceptance, D44/j9).
    When ``delta_accept == 0``, the center moves to any strictly better member
    among offspring (parent index 0 never replaces itself via ``i == 0``).

    Optional ``sigma_schedule`` supplies per-step σ; if omitted, constant
    ``sigma`` is used for every step.
    """

    def __init__(
        self,
        center_init: torch.Tensor,
        lambda_offspring: int = 2,
        sigma: float = 0.15,
        delta_accept: float = 0.0,
        lb: torch.Tensor | None = None,
        ub: torch.Tensor | None = None,
        sigma_schedule: torch.Tensor | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        device = torch.get_default_device() if device is None else device
        assert lambda_offspring >= 1
        assert sigma > 0
        assert 0.0 <= delta_accept < 1.0

        center_init = center_init.to(device=device)
        dim = int(center_init.shape[0])
        self.dim = dim
        self.lambda_offspring = lambda_offspring

        self.delta_accept = Parameter(delta_accept, device=device)
        self.sigma = Parameter(sigma, device=device)

        if lb is None:
            lb = torch.zeros(dim, device=device)
        if ub is None:
            ub = torch.ones(dim, device=device)
        lb = lb.to(device=device)
        ub = ub.to(device=device)
        self.lb = lb
        self.ub = ub

        self.center = Mutable(center_init.clone())
        self.best_f = Mutable(torch.tensor(float("inf"), device=device))
        self.best_x = Mutable(center_init.clone())
        self.step_index = Mutable(torch.tensor(0, dtype=torch.long, device=device))
        self.last_f_center = Mutable(torch.tensor(float("inf"), device=device))
        self.last_f_best = Mutable(torch.tensor(float("inf"), device=device))
        self.last_best_index = Mutable(torch.tensor(0, dtype=torch.long, device=device))

        if sigma_schedule is not None:
            sched = sigma_schedule.to(device=device).reshape(-1)
            self.sigma_schedule = Mutable(sched)
        else:
            self.sigma_schedule = Mutable(torch.empty(0, device=device))

    def _sigma_at_step(self) -> torch.Tensor:
        sched = self.sigma_schedule
        if sched.numel() == 0:
            return self.sigma
        idx = torch.clamp(self.step_index, max=sched.numel() - 1).long()
        return sched[idx]

    def step(self):
        device = self.center.device
        sig = self._sigma_at_step()
        lam = self.lambda_offspring

        noise = torch.randn(lam, self.dim, device=device) * sig
        offspring = (self.center.unsqueeze(0) + noise).clamp(self.lb, self.ub)
        population = torch.cat([self.center.unsqueeze(0), offspring], dim=0)

        fitness = self.evaluate(population)

        best_idx = torch.argmin(fitness)
        f_center = fitness[0]
        f_best = fitness[best_idx]

        delta = self.delta_accept
        improved = (best_idx != 0) & (f_best < f_center * (1.0 - delta))
        chosen = population[best_idx]
        mask = improved.unsqueeze(-1)
        self.center = torch.where(mask, chosen, self.center)
        self.best_f = torch.where(improved, f_best, self.best_f)
        self.best_x = torch.where(mask, chosen, self.best_x)

        self.last_f_center = f_center
        self.last_f_best = f_best
        self.last_best_index = best_idx

        self.step_index = self.step_index + 1

    def record_step(self):
        return {
            "center": self.center,
            "best_x": self.best_x,
            "best_f": self.best_f,
            "step_index": self.step_index,
        }

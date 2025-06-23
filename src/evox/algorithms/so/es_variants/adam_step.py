import torch


def adam_single_tensor(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    beta1: float | torch.Tensor = 0.9,
    beta2: float | torch.Tensor = 0.999,
    lr: float | torch.Tensor = 1e-3,
    weight_decay: float | torch.Tensor = 0,
    eps: float | torch.Tensor = 1e-8,
    decouple_weight_decay: bool = False,
):
    # weight decay
    # if weight_decay != 0:
    if decouple_weight_decay:
        param = param * (1 - weight_decay * lr)
    else:
        grad = grad + weight_decay * param
    # Decay the first and second moment running average coefficient
    exp_avg = torch.lerp(exp_avg, grad, 1 - beta1)
    exp_avg_sq = exp_avg_sq * beta2 + grad * grad.conj() * (1 - beta2)
    denom = exp_avg_sq.sqrt() + eps
    return param - lr * exp_avg / denom, exp_avg, exp_avg_sq

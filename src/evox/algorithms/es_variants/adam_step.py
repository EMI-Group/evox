import torch


def adam_single_tensor(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    beta1: torch.Tensor = torch.tensor(0.9),
    beta2: torch.Tensor = torch.tensor(0.999),
    lr: torch.Tensor = torch.tensor(1e-3),
    weight_decay: torch.Tensor = torch.tensor(0),
    eps: torch.Tensor = torch.tensor(1e-8),
):
    # weight decay
    if weight_decay != 0:
        weight_decay = weight_decay.to(param.device)
        grad = grad + weight_decay * param
    # Decay the first and second moment running average coefficient
    beta1 = beta1.to(param.device)
    beta2 = beta2.to(param.device)
    lr = lr.to(param.device)
    eps = eps.to(param.device)
    exp_avg = torch.lerp(exp_avg, grad, 1 - beta1)
    exp_avg_sq = exp_avg_sq * beta2 + grad * grad.conj() * (1 - beta2)
    denom = exp_avg_sq.sqrt() + eps
    return param - lr * exp_avg / denom, exp_avg, exp_avg_sq

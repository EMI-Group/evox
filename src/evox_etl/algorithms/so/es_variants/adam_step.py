"""Functional ETL port of the torch evox `adam_step.py` helper.

Plain (non-`@etl.defn`) function: it may only be called inside an active
trace (a function passed to `etl.build`/`etl.evaluate`), since ETL has no
eager mode. Semantics mirror the torch original in
`src/evox/algorithms/so/es_variants/adam_step.py` exactly.
"""

import etl

Tensor = etl.SymbolicTensor


def adam_single_tensor(
    param: Tensor,
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    beta1: float = 0.9,
    beta2: float = 0.999,
    lr: float = 1e-3,
    weight_decay: float = 0,
    eps: float = 1e-8,
    decouple_weight_decay: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """One Adam update step on a single tensor.

    Returns `(new_param, new_exp_avg, new_exp_avg_sq)`. With
    `decouple_weight_decay=True` the decay is applied to `param` directly
    (AdamW style); otherwise it is added into `grad` (L2 style), matching
    the torch reference.
    """
    # weight decay
    if decouple_weight_decay:
        param = param * (1 - weight_decay * lr)
    else:
        grad = grad + weight_decay * param
    # Decay the first and second moment running average coefficient
    exp_avg = beta1 * exp_avg + (1.0 - beta1) * grad
    exp_avg_sq = exp_avg_sq * beta2 + grad * grad * (1.0 - beta2)
    denom = etl.sqrt(exp_avg_sq) + eps
    return param - lr * exp_avg / denom, exp_avg, exp_avg_sq

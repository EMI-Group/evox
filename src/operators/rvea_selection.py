import torch
import torch.nn.functional as F
from ..utils import clamp, maximum, nanmin
from ..core import vmap, jit

# The following code defines a custom function `apd_func` and applies vectorization (vmap)
# and JIT compilation for optimization.

# def apd_func(
#     x: torch.Tensor,
#     y: torch.Tensor,
#     z: torch.Tensor,
#     obj: torch.Tensor,
#     theta: torch.Tensor,
# ):
#     return (
#         1 + obj.shape[1] * theta * torch.index_select(z, 0, torch.relu(x)) / y[None]
#     ) * torch.linalg.vector_norm(obj[x, :], dim=1)
#
#
# apd_fn = vmap(apd_func, in_dims=(1, 0, 1, None, None), out_dims=1, trace=False)
# apd_fn = jit(
#     apd_fn,
#     trace=True,
#     lazy=False,
#     example_inputs=(
#         torch.zeros(10000, 6000, dtype=torch.int64, device="cuda"),
#         torch.empty(6000, device="cuda"),
#         torch.empty(10000, 6000, device="cuda"),
#         torch.empty(10000, 3, device="cuda"),
#         torch.tensor(0.5, device="cuda"),
#     ),
# )


def apd_fn(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    obj: torch.Tensor,
    theta: torch.Tensor,
):
    selected_z = torch.gather(z, 0, torch.relu(x))
    left = (1 + obj.shape[1] * theta * selected_z) / y[None, :]
    norm_obj = torch.linalg.vector_norm(obj**2, dim=1)
    right = norm_obj[x]
    return left * right


def ref_vec_guided(
    x: torch.Tensor, f: torch.Tensor, v: torch.Tensor, theta: torch.Tensor
):
    n, m = f.shape
    nv = v.shape[0]

    obj = f - nanmin(f, dim=0, keepdim=True)[0]

    obj = maximum(obj, torch.tensor(1e-32, device=f.device))

    cosine = F.cosine_similarity(v.unsqueeze(1), v.unsqueeze(0), dim=-1)

    cosine = torch.where(
        torch.eye(cosine.shape[0], dtype=torch.bool, device=f.device),
        torch.tensor(0.0, device=f.device),
        cosine,
    )
    cosine = clamp(
        cosine, torch.tensor(0.0, device=f.device), torch.tensor(1.0, device=f.device)
    )
    gamma = torch.min(torch.acos(cosine), dim=1)[0]

    angle = torch.acos(
        clamp(
            F.cosine_similarity(obj.unsqueeze(1), v.unsqueeze(0), dim=-1),
            torch.tensor(0.0, device=f.device),
            torch.tensor(1.0, device=f.device),
        )
    )

    nan_mask = torch.isnan(obj).any(dim=1)
    associate = torch.argmin(angle, dim=1)
    associate = torch.where(nan_mask, torch.tensor(-1, device=f.device), associate)
    associate = associate[:, None]
    partition = torch.arange(0, n, device=f.device)[:, None]
    I = torch.arange(0, nv, device=f.device)[None, :]
    partition = (associate == I) * partition + (associate != I) * -1

    mask = associate != I
    mask_null = mask.sum(dim=0) == n

    apd = apd_fn(partition, gamma, angle, obj, theta)
    apd = torch.where(mask, float("inf"), apd)

    # TODO: The current RVEA selection implementation is suboptimal.
    #       We will implement a `segment_sort` or `segment_argmin` in CUDA in the future
    #       to optimize the process by skipping the partition, mask, mask_null,
    #       and directly calculating the survivors within each partition.

    next_ind = torch.argmin(apd, dim=0)
    next_x = torch.where(mask_null.unsqueeze(1), torch.nan, x[next_ind])
    next_f = torch.where(mask_null.unsqueeze(1), torch.nan, f[next_ind])

    return next_x, next_f

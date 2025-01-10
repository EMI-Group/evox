import torch
import torch.nn.functional as F

from ...utils import clamp_float, maximum, nanmin


def apd_fn(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    obj: torch.Tensor,
    theta: torch.Tensor,
):
    selected_z = torch.gather(z, 0, torch.relu(x))
    left = (1 + obj.size(1) * theta * selected_z) / y[None, :]
    norm_obj = torch.linalg.vector_norm(obj**2, dim=1)
    right = norm_obj[x]
    return left * right


def ref_vec_guided(x: torch.Tensor, f: torch.Tensor, v: torch.Tensor, theta: torch.Tensor):
    n, m = f.size()
    nv = v.size(0)

    obj = f - nanmin(f, dim=0, keepdim=True)[0]

    obj = maximum(obj, torch.tensor(1e-32, device=f.device))

    cosine = F.cosine_similarity(v.unsqueeze(1), v.unsqueeze(0), dim=-1)

    cosine = torch.where(
        torch.eye(cosine.size(0), dtype=torch.bool, device=f.device),
        0,
        cosine,
    )
    cosine = clamp_float(cosine, 0.0, 1.0)
    gamma = torch.min(torch.acos(cosine), dim=1)[0]

    angle = torch.acos(
        clamp_float(
            F.cosine_similarity(obj.unsqueeze(1), v.unsqueeze(0), dim=-1),
            0.0,
            1.0,
        )
    )

    nan_mask = torch.isnan(obj).any(dim=1)
    associate = torch.argmin(angle, dim=1)
    associate = torch.where(nan_mask, -1, associate)
    associate = associate[:, None]
    partition = torch.arange(0, n, device=f.device)[:, None]
    IndexMatrix = torch.arange(0, nv, device=f.device)[None, :]
    partition = (associate == IndexMatrix) * partition + (associate != IndexMatrix) * -1

    mask = associate != IndexMatrix
    mask_null = mask.sum(dim=0) == n

    apd = apd_fn(partition, gamma, angle, obj, theta)
    apd = torch.where(mask, torch.inf, apd)

    # TODO: The current RVEA selection implementation is suboptimal.
    #       We will implement a `segment_sort` or `segment_argmin` in CUDA in the future
    #       to optimize the process by skipping the partition, mask, mask_null,
    #       and directly calculating the survivors within each partition.

    next_ind = torch.argmin(apd, dim=0)
    next_x = torch.where(mask_null.unsqueeze(1), torch.nan, x[next_ind])
    next_f = torch.where(mask_null.unsqueeze(1), torch.nan, f[next_ind])

    return next_x, next_f

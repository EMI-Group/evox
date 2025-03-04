import torch
import torch.nn.functional as F

from evox.utils import clamp_float, maximum, nanmin


def apd_fn(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    obj: torch.Tensor,
    theta: torch.Tensor,
):
    """
    Compute the APD (Angle-Penalized Distance) based on the given inputs.

    :param x: A tensor representing the indices of the partition.
    :param y: A tensor representing the gamma.
    :param z: A tensor representing the angle.
    :param obj: A tensor of shape (n, m) representing the objectives of the solutions.
    :param theta: A tensor representing the parameter theta used for scaling the reference vector.

    :return: A tensor containing the APD values for each solution.
    """
    selected_z = torch.gather(z, 0, torch.relu(x))
    left = (1 + obj.size(1) * theta * selected_z) / y[None, :]
    norm_obj = torch.linalg.vector_norm(obj, dim=1)
    right = norm_obj[x]
    return left * right


def ref_vec_guided(x: torch.Tensor, f: torch.Tensor, v: torch.Tensor, theta: torch.Tensor):
    """
    Perform the Reference Vector Guided Evolutionary Algorithm (RVEA) selection process.

    This function selects solutions based on the Reference Vector Guided Evolutionary Algorithm.
    It calculates the distances and angles between solutions and reference vectors, and returns
    the next set of solutions to be evolved.

    :param x: A tensor of shape (n, d) representing the current population solutions.
    :param f: A tensor of shape (n, m) representing the objective values for each solution.
    :param v: A tensor of shape (r, m) representing the reference vectors.
    :param theta: A tensor representing the parameter theta used in the APD calculation.

    :return: A tuple containing:
        - next_x: The next selected solutions.
        - next_f: The objective values of the next selected solutions.

    :note:
        The function computes the distances between the solutions and reference vectors,
        and selects the solutions with the minimum APD.
        It currently uses a suboptimal selection implementation, and future improvements
        will optimize the process using a `segment_sort` or `segment_argmin` in CUDA.
    """
    n = f.size(0)
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

    next_ind = torch.argmin(apd, dim=0)
    next_x = torch.where(mask_null.unsqueeze(1), torch.nan, x[next_ind])
    next_f = torch.where(mask_null.unsqueeze(1), torch.nan, f[next_ind])

    return next_x, next_f

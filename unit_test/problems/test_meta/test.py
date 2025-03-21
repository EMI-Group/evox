import torch
from evox.utils import clamp, nanmax, nanmin

x = torch.tensor([[1.0, 2.0], [float('nan'), float('nan')]])
result = nanmax(x, dim=0)
print(result.values)  # Output: tensor([1.0, 4.0])
print(result.indices)  # Output: tensor([0, 1])
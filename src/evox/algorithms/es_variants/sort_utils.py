import torch


def sort_by_key(keys, *vals):
    assert (
        len(keys.size()) == 1
    ), f"Expect keys to be 1D tensor, got {keys.size()}."
    order = torch.argsort(keys)
    vals = map(lambda x: x[order], vals)
    
    return keys[order], *vals
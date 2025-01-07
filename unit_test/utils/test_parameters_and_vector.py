import torch
import torch.nn as nn

import os
import sys

current_directory = os.getcwd()
if current_directory not in sys.path:
    sys.path.append(current_directory)

from src.utils import ParamsAndVector


if __name__ == "__main__":

    # Test `to_vector` and `to_params` functions
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 3, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(3, 3, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Linear(108, 32), nn.ReLU(), nn.Linear(32, 10)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    model = SimpleCNN()
    adapter = ParamsAndVector(dummy_model=model)
    model_params = dict(model.named_parameters())

    flat_params = adapter.to_vector(model_params)
    restored_params = adapter.to_params(flat_params)

    x = torch.rand(size=(5, 1, 28, 28))
    model.load_state_dict(model_params)
    print("Before flattening: \n\t", model(x).sum())

    model.load_state_dict(restored_params)
    print("After restoring (the result should be the same as above): \n\t", model(x).sum())
    print()

    # Test `batched_to_vector` and `batched_to_params` functions
    class Dim0_Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.cat = nn.Parameter(torch.rand(1))
            self.mouse = nn.Parameter(torch.rand(1))
            self.dog = nn.Parameter(torch.rand(1))

    class Dim1_Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.cat = nn.Parameter(torch.rand(1, 2))
            self.mouse = nn.Parameter(torch.rand(1, 1))
            self.dog = nn.Parameter(torch.rand(1, 3))

    class Dim2_Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.cat = nn.Parameter(torch.rand(1, 2, 3))
            self.mouse = nn.Parameter(torch.rand(1, 1, 4))
            self.dog = nn.Parameter(torch.rand(1, 3, 2))

    model_groups = tuple([eval(f"Dim{i}_Model()") for i in range(0, 3)])
    BATCH_SIZE = 5

    for index, test_model in enumerate(model_groups):
        batched_params = {
            key: torch.stack([value] + [torch.randn_like(value) for _ in range(BATCH_SIZE - 1)])
            for key, value in dict(test_model.named_parameters()).items()
        }
        temp_adapter = ParamsAndVector(dummy_model=test_model)
        batched_flat_params = temp_adapter.batched_to_vector(batched_params)
        batched_restored_params = temp_adapter.batched_to_params(batched_flat_params)

        print(f"In the test of dimension {index}.")
        print("Before flattening: ")
        print("\n".join([f"{key}: {value}" for key, value in batched_params.items()]))
        print("-" * 10)
        print("After restoring (the result should be the same as above): ")
        print("\n".join([f"{key}: {value}" for key, value in batched_restored_params.items()]))
        print("-" * 10)
        print("Flattened result: \n", batched_flat_params)
        print()

import time
import os
import sys

current_directory = os.getcwd()
if current_directory not in sys.path:
    sys.path.append(current_directory)

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from src.utils import ParamsAndVector
from src.algorithms import PSO
from src.workflows import StdWorkflow, EvalMonitor
from src.problems.neuroevolution import SupervisedLearningProblem


if __name__ == "__main__":

    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 3, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(3, 3, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(108, 32),
                nn.ReLU(),
                nn.Linear(32, 10)
            )
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x
        
    
    def model_test(model, data_loader, device):
        model.eval()
        with torch.no_grad():
            total = 0
            correct = 0
            with torch.no_grad():
                for inputs, labels in data_loader:
                    inputs = inputs.to(device=device, non_blocking=True)
                    labels = labels.to(device=device, non_blocking=True)
        
                    logits = model(inputs)
                    _, predicted = torch.max(logits.data, dim=1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                acc = 100 * correct / total
        return acc
    
    
    def model_train(model, data_loader, criterion, optimizer, max_epoch, device, print_frequent=-1):
        model.train()
        for epoch in range(max_epoch):
            running_loss = 0.0
            for step, (inputs, labels) in enumerate(data_loader, start=1):
                inputs = inputs.to(device=device, non_blocking=True)
                labels = labels.to(device=device, non_blocking=True)
    
                optimizer.zero_grad()
                logits = model(inputs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
    
                running_loss += loss.item()
                if print_frequent > 0 and step % print_frequent == 0:
                    print(f"[{epoch:d}, {step:4d}] runing loss: {running_loss:.4f}")
                    running_loss = 0.0
        return model
    
    
    def neuroevolution_process(
        workflow      : StdWorkflow, 
        adapter       : ParamsAndVector, 
        model         : nn.Module, 
        test_loader   : DataLoader, 
        device        : torch.device, 
        max_generation: int = 50,
    ) -> None:
        best_acc = -1
        for index in range(max_generation):
            print(f"In generation {index}:")
            t = time.time()
            workflow.step()
            torch.cuda.synchronize()
            print(f"\tTime elapsed: {time.time() - t: .4f}(s).")
    
            monitor: EvalMonitor = workflow.get_submodule("monitor")
            print(f"\tTop 1 fitness: {monitor.topk_fitness[0]:.4f}.")
            print(f"\tTop 2 fitness: {monitor.topk_fitness[1]:.4f}.")
            best_params = adapter.to_params(monitor.topk_solutions[0])
            model.load_state_dict(best_params)
            acc = model_test(model, test_loader, device)
            if acc > best_acc:
                best_acc = acc
            print(f"\tBest accuracy: {best_acc:.4f} %.")


    # General setting
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    data_root = "./data"
    os.makedirs(data_root, exist_ok=True)

    # Set random seed
    seed = 0
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = False
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False

    # Define dataset and data loader
    BATCH_SIZE = 100
    train_dataset = torchvision.datasets.MNIST(
        root      = data_root,
        train     = True,
        download  = True,
        transform = torchvision.transforms.ToTensor(),
    )
    train_loader = DataLoader(train_dataset,
        batch_size = BATCH_SIZE,
        shuffle    = True,
        collate_fn = None,
    )
    test_dataset = torchvision.datasets.MNIST(
        root      = data_root,
        train     = False,
        download  = True,
        transform = torchvision.transforms.ToTensor(),
    )
    test_loader = DataLoader(test_dataset,
        batch_size = BATCH_SIZE,
        shuffle    = False,
        collate_fn = None,
    )

    # Data preloading
    # TODO: Add collate functions
    print("Data preloading start.")
    import tqdm
    pre_train_loader = tuple([
        (inputs.to(device), labels.type(torch.float).unsqueeze(1).repeat(1, 10).to(device))
        for inputs, labels in tqdm.tqdm(train_loader)
    ])
    pre_test_loader = tuple([
        (inputs.to(device), labels.to(device))
        for inputs, labels in tqdm.tqdm(test_loader)
    ])
    print()

    # Initialize model
    model = SimpleCNN().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of model parameters: {total_params}")
    print()

    # # Gradient descent process
    # print("Gradient descent training start.")
    # model_train(model, 
    #     data_loader    = train_loader, 
    #     criterion      = nn.CrossEntropyLoss(), 
    #     optimizer      = torch.optim.Adam(model.parameters(), lr=1e-3), 
    #     max_epoch      = 10, 
    #     device         = device,
    #     print_frequent = 500,
    # )
    # gd_acc = model_test(model, pre_test_loader, device)
    # print(f"Accuracy after gradient descent training: {gd_acc:.4f} %.")
    # print()

    print("Neuroevolution process start.")
    adapter = ParamsAndVector(dummy_model=model)
    model_params = dict(model.named_parameters())
   
    # Define criterion
    class AccuracyCriterion(nn.Module):
        def __init__(self, data_loader: DataLoader):
            super().__init__()
            self.data_loader = data_loader
        def forward(self, logits, labels):
            _, predicted = torch.max(logits, dim=1)
            correct = (predicted == labels[:, 0]).sum()
            fitness = -correct
            return fitness
    acc_criterion = AccuracyCriterion(pre_train_loader)
    loss_criterion = nn.MSELoss()
    class WeightedCriterion(nn.Module):
        def __init__(self, loss_weight, loss_criterion, acc_weight, acc_criterion):
            super().__init__()
            self.loss_weight    = loss_weight
            self.loss_criterion = loss_criterion
            self.acc_weight     = acc_weight
            self.acc_criterion  = acc_criterion
        def forward(self, logits, labels):
            weighted_loss = self.loss_weight * self.loss_criterion(logits, labels)
            weighted_acc  = self.acc_weight  * self.acc_criterion(logits, labels)
            return weighted_loss + weighted_acc
    weighted_criterion = WeightedCriterion(
        loss_weight    = 0.5,
        loss_criterion = loss_criterion,
        acc_weight     = 0.5,
        acc_criterion  = acc_criterion,
    )

    # Initialize monitor
    monitor = EvalMonitor(topk=2, device=device) # choose the best two individuals
    monitor.setup()


    # Population-based neuroevolution testing
    print("Population-based neuroevolution process start.")
    POP_SIZE = 1000
    vmapped_problem = SupervisedLearningProblem(
        model       = model,
        data_loader = pre_train_loader,
        criterion   = weighted_criterion,
        pop_size    = POP_SIZE,
        device      = device,
    )
    vmapped_problem.setup()

    pop_center = adapter.to_vector(model_params)
    pop_algorithm = PSO(
        pop_size = POP_SIZE, 
        lb       = pop_center - 0.01, 
        ub       = pop_center + 0.01, 
        device   = device,
    )
    pop_algorithm.setup()

    workflow = StdWorkflow()
    workflow.setup(
        algorithm          = pop_algorithm, 
        problem            = vmapped_problem,
        solution_transform = adapter,
        monitor            = monitor,
        device             = device,
    )
    neuroevolution_process(
        workflow       = workflow, 
        adapter        = adapter, 
        model          = model, 
        test_loader    = pre_test_loader, 
        device         = device,
        max_generation = 5,
    )
    print()


    # # Single-run neuroevolution testing
    # print("Single-run neuroevolution process start.")
    # single_problem = SupervisedLearningProblem(
    #     model       = model,
    #     data_loader = pre_train_loader,
    #     criterion   = weighted_criterion,
    #     pop_size    = None, # set the problem as single-run mode
    #     device      = device,
    # )
    # single_problem.setup()

    # # TODO: Add a single-run random search algorithm
    # single_algorithm = None

    # # Reset the `workflow` with single-run based problem and algorithm
    # workflow.setup(
    #     algorithm          = single_algorithm, 
    #     problem            = single_problem,
    #     solution_transform = adapter,
    #     monitor            = monitor,
    #     device             = device,
    # )
    # neuroevolution_process(
    #     workflow       = workflow, 
    #     adapter        = adapter, 
    #     model          = model, 
    #     test_loader    = pre_test_loader, 
    #     device         = device,
    #     max_generation = 5,
    # )
    # print()

    print("Tests completed.")

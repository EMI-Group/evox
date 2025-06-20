import os
import time
import unittest

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from evox.algorithms import PSO
from evox.problems.neuroevolution.supervised_learning import SupervisedLearningProblem
from evox.utils import ParamsAndVector
from evox.workflows import EvalMonitor, StdWorkflow


class SampleCNN(nn.Module):
    def __init__(self):
        super(SampleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(3, 3, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(3, 3, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=3),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(12, 10))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class TestSupervisedLearningProblem(unittest.TestCase):
    def setUp(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.data_root = "./data"
        os.makedirs(self.data_root, exist_ok=True)

        # Set random seed
        seed = 0
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # Define dataset and data loader
        BATCH_SIZE = 100
        self.train_dataset = torchvision.datasets.MNIST(
            root=self.data_root,
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=None,
        )
        self.test_dataset = torchvision.datasets.MNIST(
            root=self.data_root,
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=None,
        )

        # Data preloading
        self.pre_gd_train_loader = tuple(
            [(inputs.to(self.device), labels.to(self.device)) for inputs, labels in self.train_loader]
        )
        self.pre_ne_train_loader = tuple(
            [
                (
                    inputs.to(self.device),
                    labels.type(torch.float).unsqueeze(1).repeat(1, 10).to(self.device),
                )
                for inputs, labels in self.train_loader
            ]
        )
        self.pre_test_loader = tuple(
            [(inputs.to(self.device), labels.to(self.device)) for inputs, labels in self.test_loader]
        )

        self.model = SampleCNN().to(self.device)
        self.adapter = ParamsAndVector(dummy_model=self.model)
        self.model_params = dict(self.model.named_parameters())
        self.pop_center = self.adapter.to_vector(self.model_params)
        self.lower_bound = self.pop_center - 0.01
        self.upper_bound = self.pop_center + 0.01

        class AccuracyCriterion(nn.Module):
            def __init__(self, data_loader):
                super().__init__()
                self.data_loader = data_loader

            def forward(self, logits, labels):
                _, predicted = torch.max(logits, dim=1)
                correct = (predicted == labels[:, 0]).sum()
                fitness = -correct
                return fitness

        self.acc_criterion = AccuracyCriterion(self.pre_ne_train_loader)
        self.loss_criterion = nn.MSELoss()

        class WeightedCriterion(nn.Module):
            def __init__(self, loss_weight, loss_criterion, acc_weight, acc_criterion):
                super().__init__()
                self.loss_weight = loss_weight
                self.loss_criterion = loss_criterion
                self.acc_weight = acc_weight
                self.acc_criterion = acc_criterion

            def forward(self, logits, labels):
                weighted_loss = self.loss_weight * self.loss_criterion(logits, labels)
                weighted_acc = self.acc_weight * self.acc_criterion(logits, labels)
                return weighted_loss + weighted_acc

        self.weighted_criterion = WeightedCriterion(
            loss_weight=0.5,
            loss_criterion=self.loss_criterion,
            acc_weight=0.5,
            acc_criterion=self.acc_criterion,
        )

    def model_test(self, model, data_loader, device):
        model.eval()
        with torch.no_grad():
            total = 0
            correct = 0
            for inputs, labels in data_loader:
                inputs = inputs.to(device=device, non_blocking=True)
                labels = labels.to(device=device, non_blocking=True)

                logits = model(inputs)
                _, predicted = torch.max(logits.data, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            acc = 100 * correct / total
        return acc

    def model_train(
        self,
        model,
        data_loader,
        criterion,
        optimizer,
        max_epoch,
        device,
        print_frequent=-1,
    ):
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
                    print(f"[Epoch {epoch:2d}, step {step:4d}] running loss: {running_loss:.4f} ")
                    running_loss = 0.0
        return model

    def neuroevolution_process(self, workflow, adapter, model, test_loader, device, best_acc, max_generation=2):
        for index in range(max_generation):
            print(f"In generation {index}:")
            t = time.time()
            workflow.step()
            print(f"\tTime elapsed: {time.time() - t: .4f}(s).")

            monitor = workflow.get_submodule("monitor")
            print(f"\tTop fitness: {monitor.topk_fitness}")
            best_params = adapter.to_params(monitor.topk_solutions[0])
            model.load_state_dict(best_params)
            acc = self.model_test(model, test_loader, device)
            if acc > best_acc:
                best_acc = acc
            print(f"\tBest accuracy: {best_acc:.4f} %.")

    def test_gradient_descent_training(self):
        print("Gradient descent training start.")
        self.model_train(
            self.model,
            data_loader=self.pre_gd_train_loader,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(self.model.parameters(), lr=1e-2),
            max_epoch=3,
            device=self.device,
            print_frequent=500,
        )
        gd_acc = self.model_test(self.model, self.pre_test_loader, self.device)
        print(f"Accuracy after gradient descent training: {gd_acc:.4f} %.")
        self.assertGreater(gd_acc, 80.0)

    def test_population_based_neuroevolution(self):
        print("Population-based neuroevolution process start.")
        POP_SIZE = 4
        vmapped_problem = SupervisedLearningProblem(
            model=self.model,
            data_loader=self.pre_ne_train_loader,
            criterion=self.weighted_criterion,
            pop_size=POP_SIZE,
            device=self.device,
        )

        pop_algorithm = PSO(
            pop_size=POP_SIZE,
            lb=self.lower_bound,
            ub=self.upper_bound,
            device=self.device,
        )

        pop_monitor = EvalMonitor(
            topk=3,
            device=self.device,
        )

        pop_workflow = StdWorkflow(
            algorithm=pop_algorithm,
            problem=vmapped_problem,
            monitor=pop_monitor,
            solution_transform=self.adapter,
            device=self.device,
        )
        self.neuroevolution_process(
            workflow=pop_workflow,
            adapter=self.adapter,
            model=self.model,
            test_loader=self.pre_test_loader,
            device=self.device,
            best_acc=0.0,
            max_generation=3,
        )

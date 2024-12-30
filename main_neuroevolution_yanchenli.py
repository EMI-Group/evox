import time
import os
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.profiler import profile, ProfilerActivity

from src.utils import ParamsAndVector
from src.core import Problem, use_state, jit
from src.algorithms import PSO
from src.workflows import StdWorkflow, EvalMonitor
from src.problems.neuroevolution import SupervisedLearningProblem

class Print(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        print(x.shape)
        return x

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            # Print(),
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
        # print(x.size())
        x = self.classifier(x)
        return x
    

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # torch.set_default_device(device)
    data_root = "./data"
    os.makedirs(data_root, exist_ok=True)

    train_dataset = torchvision.datasets.MNIST(
        root      = data_root,
        train     = True,
        download  = True,
        transform = torchvision.transforms.ToTensor(),
    )
    train_loader = DataLoader(train_dataset,
        batch_size = 10,
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
        batch_size = 10,
        shuffle    = False,
        collate_fn = None,
    )

    import tqdm
    trainloader = [
        (inputs.to(device), labels.to(device))
        for inputs, labels in tqdm.tqdm(train_loader)
    ]
    testloader = [
        (inputs.to(device), labels.to(device))
        for inputs, labels in tqdm.tqdm(test_loader)
    ]


    model = SimpleCNN().to(device=device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    adapter = ParamsAndVector(dummy_model=model).to(device=device)
    model_params = dict(model.named_parameters())
    
    prob = SupervisedLearningProblem(data_loader=train_loader)
    prob.setup(model=model, criterion=nn.MSELoss(), device=device)
    prob.to(device)

    center = adapter.to_vector(model_params)

    algo = PSO(pop_size=23).to(device)
    algo.setup(lb=center - 10, ub=center + 10)
    monitor = EvalMonitor(topk=1) # best one
    monitor.setup()
    monitor.to(device)
    workflow = StdWorkflow()
    workflow.setup(
        algorithm          = algo, 
        problem            = prob,
        solution_transform = adapter,
        monitor            = monitor,
        device             = device
    )
    for _ in range(50):
        workflow.step()

        monitor = workflow.get_submodule("monitor")
        print(monitor.topk_fitness[0])
        best_params = adapter.to_params(monitor.topk_solutions[0])
        model.load_state_dict(best_params)
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device=device, non_blocking=True)
                labels = labels.to(device=device, non_blocking=True)
    
                logits = model(inputs)
                _, predicted = torch.max(logits.data, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Acc: {100 * correct / total} %.")


    # -----------------------------------------
    import sys; sys.exit(1)
    # -----------------------------------------

    log_root = "./tests"
    os.makedirs(log_root, exist_ok=True)

    log_file_a = os.path.join(log_root, "a.md")
    with open(log_file_a, "w") as ff:
        ff.write(workflow.step.inlined_graph.__str__())
    print(f"Please see the result log at `{log_file_a}`.")

    state_step = use_state(lambda: workflow.step)
    state = state_step.init_state()
    ## state = {k: (v if v.ndim < 1 or v.shape[0] != algo.pop_size else v[:3]) for k, v in state.items()}
    jit_state_step = jit(state_step, trace=True, example_inputs=(state,))
    state = state_step.init_state()

    log_file_b = os.path.join(log_root, "b.md")
    with open(log_file_b, "w") as ff:
        ff.write(jit_state_step.inlined_graph.__str__())
    print(f"Please see the result log at `{log_file_b}`.")

    t = time.time()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True
    ) as prof:
        # for _ in range(1000):
        #     workflow.step()
        for _ in range(1000):
            state = jit_state_step(state)
    print(prof.key_averages().table())
    torch.cuda.synchronize()
    print(time.time() - t)

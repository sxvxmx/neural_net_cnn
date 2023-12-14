import torch.nn as nn
import torch

class SimpleNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 16),
            nn.ReLU(),
            nn.Linear(16, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_relu_stack(x)
    
    
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,12,3,stride=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(12,20,5)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.lin1 = nn.Linear(in_features=1620, out_features=256)
        self.relu1 = nn.ReLU()
        self.lin2 = nn.Linear(in_features=256, out_features=256)
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.lin3 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x:torch.Tensor):
        x = self.maxpool1(self.relu3(self.conv1(x)))
        x = self.maxpool2(self.relu4(self.conv2(x)))
        x = torch.flatten(x,1)
        x = self.relu1(self.lin1(x))
        x = self.relu2(self.lin2(x))
        x = self.lin3(x)
        return x
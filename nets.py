import torch
import torch.nn as nn

class foodnet(nn.Module):
    def __init__(self):
        super(foodnet, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5, 5), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,3), stride=2),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,3), stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(40328, 1024),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        x = self.feature(input)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output
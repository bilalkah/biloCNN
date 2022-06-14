import torch
import torch.nn as nn


class CNNLaser(nn.Module):
    def __init__(self):
        super(CNNLaser, self).__init__()
        
        # input 6x13000
        self.features = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),
        )
        
        self.linear = nn.Linear(128*203, 2048)
        
    def forward(self, x):
        x = self.features(x)
        x = self.linear(x.view(x.size(0), -1))
        return x
    
    
model = CNNLaser()
x = torch.randn(1, 6, 13000)
print(model(x).shape)
        
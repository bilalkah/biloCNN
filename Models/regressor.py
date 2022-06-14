import torch
import torch.nn as nn


class Regressor(nn.Module):
    def __init__(self, in_channels = 1280, mid_channels = [256,128], out_channels = 3):
        super(Regressor, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.middle_channels = mid_channels
        self.regressor = self.create_regressor()
        
    def create_regressor(self):
        regressor = []
        in_cha = self.in_channels
        for _, middle_channels in enumerate(self.middle_channels):
            regressor.append(
                nn.Sequential(
                    nn.Linear(in_cha, middle_channels),
                    nn.Dropout(0.2),
                    nn.ReLU()
                )
            )
            in_cha = middle_channels
        
        regressor.append(
            nn.Linear(in_cha, self.out_channels)
        )
        return nn.Sequential(*regressor)
    
    def forward(self, x):    
        return self.regressor(x)
    
if __name__ == '__main__':
    model = Regressor(in_channels=1280, out_channels=3)
    x = torch.randn(4,1280)
    print(model(x).shape)
     
            


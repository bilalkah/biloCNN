from Models.regressor import Regressor
from Models.darknet19 import Darknet19
from Models.mobilenetv2 import MobileNetV2
import torch
import torch.nn as nn
import sys


class CameraCNN(nn.Module):
    def __init__(self, in_channels=6, num_classes=10):
        super(CameraCNN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.featureNet = MobileNetV2(
           in_channels=in_channels,
        )     

        self.regressor = Regressor(
            in_channels=1280,
            out_channels=num_classes
        )
        
    def forward(self, x):
        x = self.featureNet(x)
        x = self.regressor(x)
        return x
    
    
if __name__ == '__main__':
    model = CameraCNN(in_channels=6, num_classes=3600)
    x = torch.randn(4,6,224,224)
    print(model(x).shape)
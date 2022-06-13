from Models.regressor import Regressor
from Models.darknet19 import Darknet19
from Models.mobilenetv2 import MobileNetV2
import torch
import torch.nn as nn
import sys



"""
    _summary_ = Camera CNN model for pose estimation

    Returns:
        Tensor: concatenated output of translation and rotation
"""
class CameraCNN(nn.Module):
    def __init__(self, in_channels=6, translation = 3, rotation = 3, featureNet='MobileNetV2', regressor='Regressor'):
        super(CameraCNN, self).__init__()
        self.in_channels = in_channels

        self.featureNet = MobileNetV2(
        in_channels=in_channels,
        )
        
        # self.translationNet = Regressor(
        #     in_channels=1280,
        #     out_channels=2*translation,
        # )
        
        # self.rotationNet = Regressor(
        #     in_channels=1280,
        #     out_channels=2*rotation,
        # )
        
        # initialize weights with 
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # ft = self.featureNet(x)
        
        # out_translation = self.translationNet(ft)
        # out_rotation = self.rotationNet(ft)

        # # concatenate translation and rotation
        # return torch.cat((out_translation, out_rotation), dim=1)
        return self.featureNet(x)
    
    
if __name__ == '__main__':
    model = CameraCNN(in_channels=6)
    x = torch.randn(4,6,224,224)
    print(model(x).shape)
    print(model(x))
from Models.regressor import Regressor
from Models.darknet19 import Darknet19
from Models.mobilenetv2 import MobileNetV2
import torch
import torch.nn as nn
import sys


net_cfgs = [
            # conv1s
            [(32, 3)],
            ['M', (64, 3)],
            ['M', (128, 3), (64, 1), (128, 3)],
            ['M', (256, 3), (128, 1), (256, 3)],
            ['M', (512, 3), (256, 1), (512, 3), (256, 1), (512, 3)],
            # conv2
            ['M', (1024, 3), (512, 1), (1024, 3), (512, 1), (1024, 3)],
            # ------------
            # conv3
            #[(1024, 3), (1024, 3)],
            # conv4
            #[(1024, 3)],
            [(1280,1)]
        ]

"""
    _summary_ = Lidar CNN model for pose estimation

    Returns:
        Tensor: concatenated output of translation and rotation
"""
class LidarCNN(nn.Module):
    def __init__(self,in_channels=6, translation = 3, rotation = 3):
        super(LidarCNN,self).__init__()
        self.featureNet = self.create_darknet19_features()
        
        # self.translationNet = Regressor(
        #     in_channels=1280,
        #     out_channels=2*translation,
        # )
        
        # self.rotationNet = Regressor(
        #     in_channels=1280,
        #     out_channels=2*rotation,
        # )
        
        self.init_weights()
        
    def create_darknet19_features(self):
        features = []
        in_channels = 6
        for _,layers in enumerate(net_cfgs):
            for layer in layers:
                if layer == 'M':
                    features.append(nn.MaxPool1d(kernel_size=2,stride=2))
                else:
                    out_channels,kernel_size = layer
                    if kernel_size == 1:
                        features.append(nn.Conv1d(in_channels,out_channels,kernel_size,1))
                    else:
                        features.append(nn.Conv1d(in_channels,out_channels,kernel_size,1,1))
                    features.append(nn.BatchNorm1d(out_channels))
                    features.append(nn.LeakyReLU(0.1))
                    in_channels = out_channels
        features.append(nn.AdaptiveAvgPool1d((1)))
        return nn.Sequential(*features)  

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self,x):
        # self.ft = self.featureNet(x).view(x.size(0),-1)
        # translation = self.translationNet(self.ft)
        # rotation = self.rotationNet(self.ft)
        
        # return torch.cat((translation, rotation), dim=1)
        return self.featureNet(x).view(x.size(0),-1)
    
if __name__ == '__main__':
    model = LidarCNN()
    x=torch.randn(4,6,13000)
    print(model(x).shape)
    
    
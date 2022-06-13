from CameraCNN import CameraCNN
from LidarCNN import LidarCNN
from Models.regressor import Regressor

import torch
import torch.nn as nn
import sys



class FusionCNN(nn.Module):
    def __init__(
        self, 
        in_channels=6, 
        translation = 3, 
        rotation = 3, 
        LidarCNNweight=None, 
        CameraCNNweight=None,
        frozeLayers = False,
    ):
        super(FusionCNN, self).__init__()

        self.in_channels = in_channels

        self.cameraCNN = CameraCNN()
        self.lidarCNN = LidarCNN()
        # input size is batch x fusion dimension x feature dimension
        self.fusion = nn.Conv1d(
            in_channels= 2,
            out_channels= 1,
            kernel_size= 1,
        )
        
        self.translationNet = Regressor(
            in_channels=1280,
            out_channels=translation,
        )
        
        self.rotationNet = Regressor(
            in_channels=1280,
            out_channels=rotation,
        )
            
        if LidarCNNweight is not None:
            self.lidarCNN.load_state_dict(torch.load(LidarCNNweight))
        else:
            self.lidarCNN.init_weights()
            
        if CameraCNNweight is not None:
            self.cameraCNN.load_state_dict(torch.load(CameraCNNweight))
        else:
            self.cameraCNN.init_weights()
            
        if frozeLayers:
            self.froze()
        
        
    def froze(self):
        for param in self.cameraCNN.parameters():
            param.requires_grad = False
            
        for param in self.lidarCNN.parameters():
            param.requires_grad = False
            
    def unFroze(self):
        for param in self.cameraCNN.parameters():
            param.requires_grad = True
            
        for param in self.lidarCNN.parameters():
            param.requires_grad = True
        
    def forward(self,img,lidar):
        # img,lidar = data
        cameraOut = self.cameraCNN.featureNet(img)
        LidarOut = self.lidarCNN.featureNet(lidar).view(lidar.size(0),-1)
        
        # add dimension for fusion and concatenate
        cameraOut = cameraOut.unsqueeze(1)
        LidarOut = LidarOut.unsqueeze(1)
        
        # print("cameraOut: ", cameraOut.shape)
        # print("LidarOut: ", LidarOut.shape)
        
        
        fusionOut = torch.cat((cameraOut, LidarOut), dim=1)
        # print("before fusion: ", fusionOut.shape)
        fusionOut = self.fusion(fusionOut)
        # print("after fusion: ", fusionOut.shape)
        
        # reduce dimension
        fusionOut = fusionOut.squeeze(1)
        # print("drop dimension fusion: ", fusionOut.shape)
        
        translation = self.translationNet(fusionOut)
        rotation = self.rotationNet(fusionOut)
        
        # concatenate translation and rotation
        out = torch.cat((translation, rotation), dim=1)
        # print("out: ", out.shape)
        
        return out
    
    
if __name__ == '__main__':
    model = FusionCNN()
    
    img = torch.randn(2,6,224,224)
    lidar = torch.randn(2,6,13000)
    
    out = model(img, lidar)
    
    

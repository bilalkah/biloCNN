from Models.CNNCam import CNNCam
from Models.CNNLaser import CNNLaser
from Models.regressor import Regressor
import torch
import torch.nn as nn

class PaperCNN(nn.Module):
    def __init__(self):
        super(PaperCNN, self).__init__()
        self.cam = CNNCam()
        self.laser = CNNLaser()
        
        self.trans = Regressor(4096,[1024],out_channels=270)
        self.rot = Regressor(4096,[1024],112)
        
    def forward(self,img,pcd):
        img = self.cam(img)
        pcd = self.laser(pcd)
        
        #concatenate
        x = torch.cat((img,pcd),1)
        
        translation = self.trans(x)
        rotation = self.rot(x)
        
        return translation,rotation 
    
model = PaperCNN()
x = torch.randn(1,6,416,128)
y = torch.randn(1,6,13000)
t,r = model(x,y)
print(t.shape)
print(r.shape)
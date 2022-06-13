import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

# Target and Label are both tensors
# Tensor structure: [batch_size, 6], 3 for translation, 3 for rotation


class DOF6Loss(nn.Module):
    """
    0 <= alpha <= 1
    0 <= gamma <=5
    """
    def __init__(self,epsilon=1e-9,weight=None,size_average=True) -> None:
        super(DOF6Loss,self).__init__()
        self.epsilon = epsilon
        self.size_average = size_average
        self.weight = weight
        
        
    def forward(self,pred,tar):
        prediction = pred.clone()
        target = tar.clone()
        loss = torch.zeros(prediction.shape[0],dtype=torch.float)


        prediction += self.epsilon
        
        # mean square error
        prediction[:,0:3] = prediction[:,0:3]*100
        target[:,0:3] = target[:,0:3]*100
        
        prediction[:,6:9] = prediction[:,6:9]*1000
        target[:,6:9] = target[:,6:9]*1000
        
        # mse loss for translation torch.nn.Functional.mse_loss
        
        
        
        
        
        # #loss = torch.sum((prediction - target)**2,dim=1)
                

        # # calculate average loss
        # if self.size_average:
        #     loss = torch.mean(loss)
        # else:
        #     loss = torch.sum(loss)
        
        
        return loss



if __name__ == '__main__':
    model = DOF6Loss()
    x = torch.ones(1,2,6)
    y = torch.ones(1,2,6)
    print(model(x,y))
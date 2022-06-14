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
        loss_t = torch.zeros(prediction.shape[0],dtype=torch.float)
        loss_r = torch.zeros(prediction.shape[0],dtype=torch.float)
        loss_ts = torch.zeros(prediction.shape[0],dtype=torch.float)
        loss_rs = torch.zeros(prediction.shape[0],dtype=torch.float)


        prediction += self.epsilon
        
        # mean square error
        prediction[:,0:3] = prediction[:,0:3]*100
        target[:,0:3] = target[:,0:3]*100
        
        prediction[:,6:9] = prediction[:,6:9]*1000
        target[:,6:9] = target[:,6:9]*1000
        
        # mse loss for translation use F.mse_loss
        loss_t = F.mse_loss(prediction[:,0:3],target[:,0:3],reduction='mean')
        
        # loss for signs of translation
        loss_ts = F.cross_entropy(prediction[:,3:6],target[:,3:6],reduction='sum')
        
        
        # mse loss for rotation use F.mse_loss
        loss_r = F.mse_loss(prediction[:,6:9],target[:,6:9],reduction='mean')

        # crossentropy loss for sings of rotation
        loss_rs = F.cross_entropy(prediction[:,9:12],target[:,9:12],reduction='sum')
        
                
        loss = loss_t + loss_r + loss_ts + loss_rs
        
        # calculate average loss
        if self.size_average:
            loss = loss / prediction.shape[0]
        
        print(loss_t,loss_ts,loss_r,loss_rs)
        
        return loss



if __name__ == '__main__':
    model = DOF6Loss()
    x = torch.Tensor([[1,2,3,0,1,0,7,1,9,0,0,0],[1,2,3,0,1,0,7,1,9,0,0,0]])
    y = torch.Tensor([[1,2,3,0,0,0,7,0,9,1,0,1],[1,2,3,0,0,0,7,0,7,1,0,1]])
    print(model(x,y))
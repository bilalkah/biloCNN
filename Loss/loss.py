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
        
        
    def forward(self,prediction,target):
        prediction += self.epsilon
        
        #prediction shape is [batch_size, 2, 6]
        #target shape is [batch_size, 2, 6]
        
        # normalize the prediction[:,1,:] to [0,1]
        prediction[:,1,:] = F.normalize(prediction[:,1,:])
        
        for i in range(prediction.shape[0]):
            for k in range(prediction.shape[2]):
                if prediction[i,1,k] < 0.5:
                    prediction[i,0,k] = -1 * prediction[i,0,k]
        
        
        # loss = torch.zeros(pred.shape[0],dtype=torch.float) #,device='cuda' if torch.cuda.is_available() else 'cpu'
        # sys.exit()
        
        # split target into translation and rotation
        translation_target = target[:,0,0:3]
        rotation_target = target[:,0,3:6]
        
        translation_pred = prediction[:,0,0:3]
        rotation_pred = prediction[:,0,3:6]
        
        # convert translation meter to centimeter
        translation_target = translation_target * 100
        translation_pred = translation_pred * 100
        
        # convert rotation radiant to milliradian
        rotation_target = rotation_target * 1000
        rotation_pred = rotation_pred * 1000
        
        # MSE for translation
        translation_loss = F.mse_loss(translation_pred,translation_target)
        
        # MSE for rotation
        rotation_loss = F.mse_loss(rotation_pred,rotation_target)
        
        loss = translation_loss + rotation_loss
        
        # calculate average loss
        if self.size_average:
            loss = torch.mean(loss)
        else:
            loss = torch.sum(loss)
        return loss

        
        # # total displacement in translation_pred
        # translation_pred = torch.square(translation_pred)
        # translation_pred = torch.sum(translation_pred,dim=1)
        # # print(translation_pred.shape)
        # # print(translation_pred)

        # # total displacement in translation
        # translation *= 100
        # translation = torch.square(translation)
        # translation = torch.sum(translation,dim=1)
        # # print(translation.shape)
        # # print(translation)

        # # calculate MSE loss (alpha, gamma) for translation
        # # print(translation.shape)
        # # print(translation_pred.shape)
        # translation_loss = torch.sum(torch.pow(translation - translation_pred,2),dim=0)
        # # print(translation_loss)
        
        # # translation_loss = torch.pow(1-translation_pred,self.gamma) * torch.log(translation_pred)
        
        # # calculate MSE loss (alpha, gamma) for rotation
        # rotation *= 1000
        # rotation_loss = torch.sum(torch.pow(rotation - rotation_pred,2),dim=1)
        
        # rotation_loss = torch.pow(1-rotation_pred,self.gamma) * torch.log(rotation_pred)
        
        # # calculate total loss
        # loss = translation_loss + rotation_loss
        
        # # loss = -(self.alpha * translation_loss + (1-self.alpha) * rotation_loss)
        
        # # calculate average loss
        # if self.size_average:
        #     loss = torch.mean(loss)
        # else :
        #     loss = torch.sum(loss)
        # return loss



# class DOF6Loss(nn.Module):
#     """
#     0 <= alpha <= 1
#     0 <= gamma <=5
#     """
#     def __init__(self,alpha=0.5,gamma=2,epsilon=1e-9,rotation_coef=6,translation_coef=3,weight=None,size_average=True) -> None:
#         super(DOF6Loss,self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.size_average = size_average
#         self.rotation_coef = rotation_coef
#         self.translation_coef = translation_coef

#     def forward(self,prediction,target):
#         pred = torch.clone(torch.tanh(prediction))
        
#         pred += self.epsilon
#         loss = torch.zeros(pred.shape[0],dtype=torch.float) #,device='cuda' if torch.cuda.is_available() else 'cpu'
        
#         # split target into translation and rotation
#         translation = target[:,:3]
#         rotation = target[:,3:]
        
#         # split prediction into translation and rotation
#         translation_pred = pred[:,:3]
#         rotation_pred = pred[:,3:]
        
#         # multiply translation_pred's variables by translation_coef
#         translation_pred = translation_pred * self.translation_coef
        
#         # multiply rotation_pred's variables by rotation_coef
#         rotation_pred = rotation_pred * self.rotation_coef
        
#         # total displacement in translation_pred
#         translation_pred = torch.square(translation_pred,dim=1)
#         translation_pred = torch.sum(translation_pred,dim=1)
#         print(translation_pred.shape)
#         print(translation_pred)

#         # total displacement in translation
#         translation = torch.square(translation,dim=1)
#         translation = torch.sum(translation,dim=1)
        
#         # calculate MSE loss (alpha, gamma) for translation
#         # print(translation.shape)
#         # print(translation_pred.shape)
#         translation_loss = torch.sum(torch.pow(translation - translation_pred,2),dim=1)
        
#         # translation_loss = torch.pow(1-translation_pred,self.gamma) * torch.log(translation_pred)
        
#         # calculate MSE loss (alpha, gamma) for rotation
#         rotation_loss = torch.sum(torch.pow(rotation - rotation_pred,2),dim=1)
        
#         # rotation_loss = torch.pow(1-rotation_pred,self.gamma) * torch.log(rotation_pred)
        
#         # calculate total loss
#         loss = translation_loss + rotation_loss
        
#         # loss = -(self.alpha * translation_loss + (1-self.alpha) * rotation_loss)
        
#         # calculate average loss
#         if self.size_average:
#             loss = torch.mean(loss)
#         else :
#             loss = torch.sum(loss)
#         return loss
        
        
        
#         # # distance between prediction and target
#         # distance = torch.norm(pred-target,dim=1)
        
        
        
#         # for i, p in enumerate(pred):
#         #     tar = torch.zeros(pred.shape[1],dtype=torch.float,device='cuda' if torch.cuda.is_available() else 'cpu')
#         #     tar[target[i]] = 1.
#         #     loss[i] = -torch.mean(
#         #         tar*self.alpha*torch.pow((1-p),self.gamma)*torch.log2(p) 
#         #         + 
#         #         (1-tar)*self.alpha*torch.pow((p),self.gamma)*torch.log2(1-p)
#         #     )
#         # return torch.mean(loss)


if __name__ == '__main__':
    model = DOF6Loss()
    x = torch.zeros(4,6)
    y = torch.ones(4,6)
    print(model(x,y))
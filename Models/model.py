import torch
import numpy as np
import tqdm
from time import sleep
import torchmetrics

class Model():
    def __init__(self,model,device):
        self.model = model
        self.device = device
        self.model.to(device)
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []
        self.metric = torchmetrics.Accuracy().to(device)
        
    def train(self,train_loader,val_loader,epochs,batch_size,lr,optimizer,criterion,save_name):
        for epoch in range(epochs):
            with tqdm.tqdm(train_loader,unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                epoch_loss = []
                epoch_acc = []
                
                for data, target in tepoch:
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    scores = self.model(data)
            
                    loss = criterion(scores, target)
                    if torch.isnan(loss):
                        continue
                    train_acc = self.metric(scores,target)        
                    epoch_loss.append(loss.item())
                    epoch_acc.append(train_acc.item())
                    val_loss = 0
                        
                    loss.backward()
                    optimizer.step()
            
                    tepoch.set_postfix(Train_loss = loss.item(),Train_acc = train_acc.item())
                    sleep(0.1)

                epoch_val_loss = []
                epoch_val_acc = []
                
                for _ ,(data, target) in enumerate(val_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    scores = self.model(data)
                    loss = criterion(scores, target)
                    acc = self.metric(scores,target) 
                    epoch_val_loss.append(loss.item())
                    epoch_val_acc.append(acc.item())
                    
                self.val_loss.append(np.mean(epoch_val_loss))
                self.val_acc.append(np.mean(epoch_val_acc))
                self.train_loss.append(np.mean(epoch_loss))
                self.train_acc.append(np.mean(epoch_acc))
                
                if self.val_acc[len(self.val_acc)-1] > 0.5 and np.argmax(np.asarray(self.val_acc)) == len(self.val_acc)-1:
                    self.save_weight(save_name+str(self.val_acc[-1]))
                print(f"train_loss: {np.mean(epoch_loss)} train_acc: {np.mean(epoch_acc)} - val_loss: {np.mean(epoch_val_loss)} val_acc: {np.mean(epoch_val_acc)}")
    
    def eval(self,test_loader):
        with tqdm.tqdm(test_loader,unit="batch") as tepoch:
            tepoch.set_description(f"Evaluate for test")
            epoch_acc = []
            for data, target in tepoch:
            
                data, target = data.to(self.device), target.to(self.device)
                scores = self.model(data)
            

                train_acc = self.metric(scores,target)        
                epoch_acc.append(train_acc.item())  
                

            self.test_acc = np.mean(epoch_acc)
            tepoch.set_postfix(accuracy = train_acc.item())
            sleep(0.1)
            print(f"Evaluate avg acc: {sum(epoch_acc)/len(epoch_acc)}")
    
    def save_weight(self,name):
        torch.save(self.model.state_dict(),"weights/"+name+".pth")
    
    def load_weight(self,name):
        self.model.load_state_dict(torch.load("weights/"+name+".pth"))
        
    def get_train_data(self):
        return (self.train_loss,self.train_acc)

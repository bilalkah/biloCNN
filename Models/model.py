import torch
import numpy as np
import tqdm
from time import sleep
import os

class Model():
    def __init__(self,model,device,folder="weights/"):
        self.model = model
        self.device = device
        self.model.to(device)
        self.train_loss = []
        self.val_loss = []
        self.folder = folder
        
        
    def train(self,train_loader,val_loader,epochs,optimizer,criterion,save_name,sequence):
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        self.file = open(save_name+".txt","w")
        self.file.writelines(f"{sequence}\n")
        
        self.val_loss.append(0)
        for epoch in range(epochs):
            with tqdm.tqdm(train_loader,unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                epoch_loss = []
                
                for img,pcd, target in tepoch:
                    img,pcd, target = img.to(self.device),pcd.to(self.device), target.to(self.device)

                    optimizer.zero_grad()
                    scores = self.model(img,pcd)
                    loss = criterion(scores, target)
                    if torch.isnan(loss):
                        continue
                    
                    epoch_loss.append(loss.item())
                    
                    loss.backward()
                    optimizer.step()
            
                    tepoch.set_postfix(T = loss.item(),V=self.val_loss[-1])
                    sleep(0.1)

                epoch_val_loss = []
                
                with torch.no_grad():
                    for _ ,(img,pcd, target) in enumerate(val_loader):
                        img,pcd, target = img.to(self.device),pcd.to(self.device), target.to(self.device)
                        scores = self.model(img,pcd)
                        loss = criterion(scores, target) 
                        epoch_val_loss.append(loss.item())
                    
                self.val_loss.append(np.mean(epoch_val_loss))
                self.train_loss.append(np.mean(epoch_loss))
                # tepoch.set_postfix(T_loss = self.train_loss[-1], V_loss=self.val_loss[-1])
                self.file.writelines(f"{epoch} {self.train_loss[-1]} {self.val_loss[-1]}\n")
                self.save_weight(name=save_name+"_"+sequence+"_"+str(self.train_loss[-1]))
            scheduler.step()
        self.file.close()
                
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
        torch.save(self.model.state_dict(),self.folder+name+".pth")
    
    def load_weight(self,name):
        self.model.load_state_dict(torch.load(self.folder+name+".pth"))
        
    def get_train_data(self):
        return (self.train_loss,self.train_acc)
    
    

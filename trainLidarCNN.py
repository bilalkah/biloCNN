from Models.model import Model
from LidarCNN import LidarCNN
from Dataset.dataset import KittiPCLDataset
from Loss.loss import DOF6Loss

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils import data
from torch.utils.data import DataLoader

# torch.cuda.empty_cache()
# torch.cuda.memory_summary(device=None,abbreviated=False)

batch = 32
epoch = 30

train_dataset10 = KittiPCLDataset(sequence="10")
train_loader10 = DataLoader(
    dataset=train_dataset10,
    batch_size=batch,
)

train_dataset09 = KittiPCLDataset(sequence="09")
train_loader09 = DataLoader(
    dataset=train_dataset09,
    batch_size=batch,
)

train_dataset08 = KittiPCLDataset(sequence="08")
train_loader08 = DataLoader(
    dataset=train_dataset08,
    batch_size=batch,
)

train_dataset07 = KittiPCLDataset(sequence="07")
train_loader07 = DataLoader(
    dataset=train_dataset07,
    batch_size=batch,
)

train_dataset06 = KittiPCLDataset(sequence="06")
train_loader06 = DataLoader(
    dataset=train_dataset06,
    batch_size=batch,
)

train_dataset05 = KittiPCLDataset(sequence="05")
train_loader05 = DataLoader(
    dataset=train_dataset05,
    batch_size=batch,
)

train_dataset04 = KittiPCLDataset(sequence="04")
train_loader04 = DataLoader(
    dataset=train_dataset04,
    batch_size=batch,
)

train_dataset03 = KittiPCLDataset(sequence="03")
train_loader03 = DataLoader(
    dataset=train_dataset03,
    batch_size=batch,
)

train_dataset02 = KittiPCLDataset(sequence="02")
train_loader02 = DataLoader(
    dataset=train_dataset02,
    batch_size=batch,
)

if __name__ == '__main__':
    model = Model(
        LidarCNN(in_channels=6),
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    model.folder="lidar_weights/"
    # model.load_weight("CameraCNN_04_0.08176152496159089")
    opti = optim.Adam(model.model.parameters(), lr=0.001)
    model.train_loss = []
    model.train(train_loader02,train_loader02,epochs=epoch,batch_size=batch,lr=0.001,optimizer=optim.Adam(model.model.parameters(), lr=0.001),criterion=DOF6Loss(size_average=True),save_name='LidarCNN',sequence="02")
    model.train_loss = []
    model.train(train_loader03,train_loader03,epochs=epoch,batch_size=batch,lr=0.003,optimizer=optim.Adam(model.model.parameters(), lr=0.003),criterion=DOF6Loss(size_average=True),save_name='LidarCNN',sequence="03")
    model.train_loss = []
    model.train(train_loader04,train_loader04,epochs=epoch,batch_size=batch,lr=0.002,optimizer=optim.Adam(model.model.parameters(), lr=0.002),criterion=DOF6Loss(size_average=True),save_name='LidarCNN',sequence="04")
    model.train_loss = []
    model.train(train_loader05,train_loader05,epochs=epoch,batch_size=batch,lr=0.001,optimizer=optim.Adam(model.model.parameters(), lr=0.001),criterion=DOF6Loss(size_average=True),save_name='LidarCNN',sequence="05")
    model.train_loss = []
    model.train(train_loader06,train_loader06,epochs=epoch,batch_size=batch,lr=0.003,optimizer=optim.Adam(model.model.parameters(), lr=0.003),criterion=DOF6Loss(size_average=True),save_name='LidarCNN',sequence="06")
    model.train_loss = []
    model.train(train_loader07,train_loader07,epochs=epoch,batch_size=batch,lr=0.002,optimizer=optim.Adam(model.model.parameters(), lr=0.002),criterion=DOF6Loss(size_average=True),save_name='LidarCNN',sequence="07")
    model.train_loss = []
    model.train(train_loader08,train_loader08,epochs=epoch,batch_size=batch,lr=0.001,optimizer=optim.Adam(model.model.parameters(), lr=0.001),criterion=DOF6Loss(size_average=True),save_name='LidarCNN',sequence="08")
    model.train_loss = []
    model.train(train_loader09,train_loader09,epochs=epoch,batch_size=batch,lr=0.003,optimizer=optim.Adam(model.model.parameters(), lr=0.003),criterion=DOF6Loss(size_average=True),save_name='LidarCNN',sequence="09")
    model.train_loss = []
    model.train(train_loader10,train_loader10,epochs=epoch,batch_size=batch,lr=0.002,optimizer=optim.Adam(model.model.parameters(), lr=0.002),criterion=DOF6Loss(size_average=True),save_name='LidarCNN',sequence="10")
    model.train_loss = []
    model.save_weight('LidarCNN')


from Models.model import Model
from CameraCNN import CameraCNN
from Dataset.dataset import KittiIMGDataset
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

train_dataset10 = KittiIMGDataset(sequence="10")
train_loader10 = DataLoader(
    dataset=train_dataset10,
    batch_size=batch,
)

train_dataset09 = KittiIMGDataset(sequence="09")
train_loader09 = DataLoader(
    dataset=train_dataset09,
    batch_size=batch,
)

train_dataset08 = KittiIMGDataset(sequence="08")
train_loader08 = DataLoader(
    dataset=train_dataset08,
    batch_size=batch,
)

train_dataset07 = KittiIMGDataset(sequence="07")
train_loader07 = DataLoader(
    dataset=train_dataset07,
    batch_size=batch,
)

train_dataset06 = KittiIMGDataset(sequence="06")
train_loader06 = DataLoader(
    dataset=train_dataset06,
    batch_size=batch,
)

train_dataset05 = KittiIMGDataset(sequence="05")
train_loader05 = DataLoader(
    dataset=train_dataset05,
    batch_size=batch,
)

train_dataset04 = KittiIMGDataset(sequence="04")
train_loader04 = DataLoader(
    dataset=train_dataset04,
    batch_size=batch,
)

train_dataset03 = KittiIMGDataset(sequence="03")
train_loader03 = DataLoader(
    dataset=train_dataset03,
    batch_size=batch,
)

train_dataset02 = KittiIMGDataset(sequence="02")
train_loader02 = DataLoader(
    dataset=train_dataset02,
    batch_size=batch,
)

if __name__ == '__main__':
    model = Model(
        CameraCNN(in_channels=6),
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    model.folder("camera_weights/")
    # model.load_weight("CameraCNN_04_0.08176152496159089")
    opti = optim.Adam(model.model.parameters(), lr=0.001)
    model.train_loss = []
    model.train(train_loader02,train_loader02,epochs=epoch,batch_size=batch,lr=0.001,optimizer=optim.Adam(model.model.parameters(), lr=0.001),criterion=DOF6Loss(size_average=True),save_name='CameraCNN',sequence="02")
    model.train_loss = []
    model.train(train_loader03,train_loader03,epochs=epoch,batch_size=batch,lr=0.003,optimizer=optim.Adam(model.model.parameters(), lr=0.003),criterion=DOF6Loss(size_average=True),save_name='CameraCNN',sequence="03")
    model.train_loss = []
    model.train(train_loader04,train_loader04,epochs=epoch,batch_size=batch,lr=0.002,optimizer=optim.Adam(model.model.parameters(), lr=0.002),criterion=DOF6Loss(size_average=True),save_name='CameraCNN',sequence="04")
    model.train_loss = []
    model.train(train_loader05,train_loader05,epochs=epoch,batch_size=batch,lr=0.001,optimizer=optim.Adam(model.model.parameters(), lr=0.001),criterion=DOF6Loss(size_average=True),save_name='CameraCNN',sequence="05")
    model.train_loss = []
    model.train(train_loader06,train_loader06,epochs=epoch,batch_size=batch,lr=0.003,optimizer=optim.Adam(model.model.parameters(), lr=0.003),criterion=DOF6Loss(size_average=True),save_name='CameraCNN',sequence="06")
    model.train_loss = []
    model.train(train_loader07,train_loader07,epochs=epoch,batch_size=batch,lr=0.002,optimizer=optim.Adam(model.model.parameters(), lr=0.002),criterion=DOF6Loss(size_average=True),save_name='CameraCNN',sequence="07")
    model.train_loss = []
    model.train(train_loader08,train_loader08,epochs=epoch,batch_size=batch,lr=0.001,optimizer=optim.Adam(model.model.parameters(), lr=0.001),criterion=DOF6Loss(size_average=True),save_name='CameraCNN',sequence="08")
    model.train_loss = []
    model.train(train_loader09,train_loader09,epochs=epoch,batch_size=batch,lr=0.003,optimizer=optim.Adam(model.model.parameters(), lr=0.003),criterion=DOF6Loss(size_average=True),save_name='CameraCNN',sequence="09")
    model.train_loss = []
    model.train(train_loader10,train_loader10,epochs=epoch,batch_size=batch,lr=0.002,optimizer=optim.Adam(model.model.parameters(), lr=0.002),criterion=DOF6Loss(size_average=True),save_name='CameraCNN',sequence="10")
    model.train_loss = []
    model.save_weight('CameraCNN')


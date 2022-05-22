from Models.model import Model
from CameraCNN import CameraCNN
from Dataset.dataset import KittiDataset
from Loss.loss import DOF6Loss

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils import data
from torch.utils.data import DataLoader

torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None,abbreviated=False)

train_dataset = KittiDataset(sequence="00")
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
)

if __name__ == '__main__':
    model = Model(
        CameraCNN(in_channels=6),
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    opti = optim.Adam(model.model.parameters(), lr=0.001)
    model.train(train_loader,train_loader,epochs=10,batch_size=32,lr=0.001,optimizer=opti,criterion=DOF6Loss(size_average=True),save_name='CameraCNN')


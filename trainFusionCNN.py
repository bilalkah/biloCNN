from Models.model import Model
from FusionCNN import FusionCNN
from Dataset.dataset import *
from Loss.loss import *

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils import data
from torch.utils.data import DataLoader

torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None,abbreviated=False)

batch = 32
epoch = 30

train_dataset = KittiAllDataset(sequence=["02","03","04","05","06","07","08","09","10"])
# train_dataset = KittiIMGDataset("03")
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch,
    shuffle=True,
    num_workers=4
)

validation_dataset = KittiDataset("01")
val_loader = DataLoader(
    dataset=validation_dataset,
    batch_size=batch//2,
    num_workers=4
)


# torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    model = Model(
        FusionCNN(
            in_channels=6,
            LidarCNNweight=None,
            CameraCNNweight=None,
            frozeLayers=False,
            ),
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    model.folder="fusion_weights/"
    
    
    opti = optim.Adam(
        model.model.parameters(), 
        lr=0.001,
        weight_decay=0.0001,
    )
    

    model.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epoch,
        optimizer=opti,
        criterion=DOF6LossBase(size_average=True),
        save_name='FusionCNN',
        sequence="all"
    )
    
    model.save_weight('FusionCNN')


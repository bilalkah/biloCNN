from Models.model import Model
from CameraCNN import CameraCNN
from Dataset.dataset import KittiIMGDataset, KittiIMGAllDataset
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

batch = 32
epoch = 30

# train_dataset = KittiIMGAllDataset(sequence=["02","03","04","05","06","07","08","09","10"])
train_dataset = KittiIMGDataset("03")
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch,
    shuffle=True,
    num_workers=4
)

validation_dataset = KittiIMGDataset("01")
val_loader = DataLoader(
    dataset=validation_dataset,
    batch_size=batch//2,
    num_workers=4
)


# torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    model = Model(
        CameraCNN(in_channels=6),
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    model.folder="camera_weights/"
    opti = optim.Adam(model.model.parameters(), lr=0.001)

    model.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epoch,
        batch_size=batch,
        lr=0.001,
        optimizer=optim.Adam(model.model.parameters(), lr=0.001),
        criterion=DOF6Loss(size_average=True),
        save_name='CameraCNN',
        sequence="all"
    )
    
    model.save_weight('CameraCNN')


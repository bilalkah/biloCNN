import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import glob
import numpy as np

# Dataset class for loading data
# This class is used to load data from the dataset
# and return the data in the form of tensor
# Data is image and label is pose information in txt file
# Data will be consecutive images and label will be calculated as the difference between consecutive poses

class KittiDataset(Dataset):
    def __init__(self, sequence="00", transform=None, data_augementation=False):
        self.sequence = sequence
        self.transform = transform
        self.data_augementation = data_augementation
        
        self.image_path = "/home/bilal/Desktop/project/data_odometry_color/dataset/sequences/" + sequence + "/image_2/*.png"
        self.poses_path = "/home/bilal/Desktop/project/data_odometry_poses/dataset/poses/" + sequence + ".txt"
        
        self.data = []
        self.label = []
        
        self.homogenous_matrix = np.zeros((4, 4), dtype=np.float)
        self.rotation_matrix = np.zeros((3, 3), dtype=np.float)
        self.translation_matrix = np.zeros((3, 1), dtype=np.float)
        
        self.load_data()

    def load_data(self):
        file = open(self.poses_path, "r")
        
        for images in sorted(glob.glob(self.image_path)):
            line = file.readline()
            line = line.split(" ")
            
            for i in range(3):
                for j in range(4):
                    self.homogenous_matrix[i][j] = line[i*4+j]
            
            for i in range(3):
                for j in range(3):
                    self.rotation_matrix[i][j] = self.homogenous_matrix[i][j]
            
            for i in range(3):
                self.translation_matrix[i][0] = self.homogenous_matrix[i][3]
            
            
            euler_angles = cv2.Rodrigues(self.rotation_matrix)[0]
            # print(euler_angles.shape)
            # concatenate translation and rotation
            self.label.append(np.concatenate((self.translation_matrix, euler_angles), axis=0))
            # print(np.concatenate((self.translation_matrix, euler_angles)).shape)
            self.data.append(images)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self.data):
            raise IndexError("Index out of range")
        elif idx < 0:
            raise IndexError("Index out of range")
        elif idx == len(self.data) - 1:
            idx = len(self.data) - 2
        
        img1 = cv2.imread(self.data[idx])
        img2 = cv2.imread(self.data[idx+1])
        
        # concatenate images
        img = np.concatenate((img1, img2), axis=2)
        img = cv2.resize(img, (224, 224))
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32)
        img = img / 255.0
        img = torch.from_numpy(img)
        
        label1 = self.label[idx]
        label2 = self.label[idx+1]
        
        # subtract label2 from label1
        label = label2 - label1
        
        # drop last dimension
        label = label[:,0]
        label = torch.from_numpy(label)
        return img, label
        
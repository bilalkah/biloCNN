import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import glob
import numpy as np
from open3d import *
import struct

# import pointcloud library


# Dataset class for loading data
# This class is used to load data from the dataset
# and return the data in the form of tensor
# Data is image and label is pose information in txt file
# Data will be consecutive images and label will be calculated as the difference between consecutive poses

class KittiIMGDataset(Dataset):
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
        
        
# Dataset class for loading velodyne pointcloud data to tensor    
# convert pointcloud data to tensor
# data is stored in binary file
class KittiPCLDataset(Dataset):
    def __init__(self, sequence="00", range = 120, transform=None, data_augementation=False):
        self.sequence = sequence
        self.range = range
        self.transform = transform
        self.data_augementation = data_augementation
        
        self.velodyne_path = "/home/bilal/Desktop/project/velodyne_points/data/*.bin"
        self.poses_path = "/home/bilal/Desktop/project/data_odometry_poses/dataset/poses/" + sequence + ".txt"
        
        self.data = []
        self.label = []
        
        self.load_data()
    
    def load_data(self):
        for velodyne in sorted(glob.glob(self.velodyne_path)):
            self.data.append(velodyne)
            
    def __len__(self):
        return len(self.data)
    
    
    # convert pointcloud data to tensor
    # pointcloud is 64 channel tensor
    # pointcloud is stored in binary file
    # pointcloud is stored in the form of (x, y, z, intensity)
    # x, y, z are stored in float32
    # reduce the pointcloud to 64 channel
    # thin the pointcloud to 1/10
    # drop the intensity
    def __getitem__(self, index):
        if index >= len(self.data):
            raise IndexError("Index out of range")
        elif index < 0:
            raise IndexError("Index out of range")
        elif index == len(self.data) - 1:
            index = len(self.data) - 2
        
        velodyne1 = self.data[index]
        velodyne2 = self.data[index+1]
        
        size_float = 4
        list_pcd1 = []
        with open(velodyne1, "rb") as f:
            byte = f.read(size_float * 4)
            while byte:
                x,y,z,intensity = struct.unpack("ffff", byte)
                list_pcd1.append([x,y,z])
                byte = f.read(size_float * 4)
        np_pcd = np.asarray(list_pcd1)
        pcd1 = geometry.PointCloud()
        pcd1.points = utility.Vector3dVector(np_pcd)
        
        list_pcd2 = []
        with open(velodyne2, "rb") as f:
            byte = f.read(size_float * 4)
            while byte:
                x,y,z,intensity = struct.unpack("ffff", byte)
                list_pcd2.append([x,y,z])
                byte = f.read(size_float * 4)
        np_pcd = np.asarray(list_pcd2)
        pcd2 = geometry.PointCloud()
        pcd2.points = utility.Vector3dVector(np_pcd)
        
        pcd1 = pcd1.voxel_down_sample(voxel_size=0.48)
        pcd2 = pcd2.voxel_down_sample(voxel_size=0.48)
        
        # convert pointcloud to numpy array
        pcd1 = np.asarray(pcd1.points, dtype=np.float32)
        pcd2 = np.asarray(pcd2.points, dtype=np.float32)
        print(pcd1.shape)
        print(pcd2.shape)
        
        # normalize pointcloud
        pcd1 = pcd1 / self.range
        pcd2 = pcd2 / self.range
        
        
        # resize pointcloud to 13000,3
        pcd1 = pcd1[:13000,:]
        pcd2 = pcd2[:13000,:]
        pcd1 = np.transpose(pcd1, (1, 0))
        pcd2 = np.transpose(pcd2, (1, 0))
        print(pcd1.shape)
        print(pcd2.shape)

        # # concatenate pointcloud
        pcd = np.concatenate((pcd1, pcd2), axis=0)
        pcd = torch.from_numpy(pcd)
        print(pcd.shape)
        
        return pcd
        
            
if __name__ == "__main__":
    dataset = KittiPCLDataset()
    img = dataset[1]
    print(img.shape)
    
    
        
    
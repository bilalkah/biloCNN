import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import glob
import numpy as np
from open3d import *
import struct
import pykitti

# import pointcloud library


# Dataset class for loading data
# This class is used to load data from the dataset
# and return the data in the form of tensor
# Data is image and label is pose information in txt file
# Data will be consecutive images and label will be calculated as the difference between consecutive poses

class KittiIMGDataset(Dataset):
    def __init__(self, sequence="00", path = "/home/plnm/biloCNN/kitti"):
        self.sequence = sequence
        self.path = path
        
        self.dataset = pykitti.odometry(self.path, self.sequence)
        
        self.poses = []
        
        self.homogenous_matrix = np.zeros((4, 4), dtype=np.float)
        self.rotation_matrix = np.zeros((3, 3), dtype=np.float)
        self.translation_matrix = np.zeros((3, 1), dtype=np.float)
        
        self.load_poses()

    def load_poses(self):
        
        for line in self.dataset.poses:
            for i in range(3):
                for j in range(4):
                    self.homogenous_matrix[i][j] = line[i*4+j]
            
            for i in range(3):
                for j in range(3):
                    self.rotation_matrix[i][j] = self.homogenous_matrix[i][j]
            
            for i in range(3):
                self.translation_matrix[i][0] = self.homogenous_matrix[i][3]
                
            euler_angles = cv2.Rodrigues(self.rotation_matrix)[0]
            self.poses.append(np.concatenate((self.translation_matrix, euler_angles), axis=0))

        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if idx >= len(self.dataset):
            raise IndexError("Index out of range")
        elif idx < 0:
            raise IndexError("Index out of range")
        elif idx == len(self.dataset) - 1:
            idx = len(self.dataset) - 2
        
        img1 = cv2.imread(self.dataset.cam2_files[idx])
        img2 = cv2.imread(self.dataset.cam2_files[idx+1])
        
        # concatenate images
        img = np.concatenate((img1, img2), axis=2)
        img = cv2.resize(img, (224, 224))
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32)
        img = img / 255.0
        img = torch.from_numpy(img)
        
        pose1 = self.poses[idx]
        pose2 = self.poses[idx+1]
        
        # subtract label2 from label1
        pose = pose2 - pose1
        
        # drop last dimension
        pose = pose[:,0]
        pose = torch.from_numpy(pose)
        return img, pose
        
        
# Dataset class for loading velodyne pointcloud data to tensor    
# convert pointcloud data to tensor
# data is stored in binary file
class KittiPCLDataset(Dataset):
    def __init__(self, sequence="00", max_range = 120, path = "/home/plnm/biloCNN/kitti"):
        self.sequence = sequence
        self.range = max_range
        
        self.dataset = pykitti.odometry(self.path, self.sequence)

        self.poses = []
        
        self.load_poses()
    
    def load_poses(self):
        for line in self.dataset.poses:
            for i in range(3):
                for j in range(4):
                    self.homogenous_matrix[i][j] = line[i*4+j]
            
            for i in range(3):
                for j in range(3):
                    self.rotation_matrix[i][j] = self.homogenous_matrix[i][j]
            
            for i in range(3):
                self.translation_matrix[i][0] = self.homogenous_matrix[i][3]
                
            euler_angles = cv2.Rodrigues(self.rotation_matrix)[0]
            self.poses.append(np.concatenate((self.translation_matrix, euler_angles), axis=0))
    
    def __len__(self):
        return len(self.dataset)
    
    
    # convert pointcloud data to tensor
    # pointcloud is 64 channel tensor
    # pointcloud is stored in binary file
    # pointcloud is stored in the form of (x, y, z, intensity)
    # x, y, z are stored in float32
    # reduce the pointcloud to 64 channel
    # thin the pointcloud to 1/10
    # drop the intensity
    def __getitem__(self, index):
        if index >= len(self.dataset):
            raise IndexError("Index out of range")
        elif index < 0:
            raise IndexError("Index out of range")
        elif index == len(self.dataset) - 1:
            index = len(self.dataset) - 2
        
        # get pointcloud data
        pcl1 = self.dataset.get_velo(index)[:,:3]
        pcl2 = self.dataset.get_velo(index+1)[:,:3]
        
        # convert pcd to pointcloud
        pcd1 = geometry.PointCloud()
        pcd1.points = utility.Vector3dVector(pcl1)
        
        pcd2 = geometry.PointCloud()
        pcd2.points = utility.Vector3dVector(pcl2)
        
        # compress the pointcloud to 13000-14000 points
        # for pcd1
        coeff = 0.4
        temp_pcd = pcd1.voxel_down_sample(voxel_size=coeff)
        size = np.asarray(temp_pcd.points, dtype=np.float32).shape[0]
        while size < 13000 or size > 14000:
            if size < 13000:
                coeff = coeff - 0.01
            else:
                coeff = coeff + 0.01
            temp_pcd = pcd1.voxel_down_sample(voxel_size=coeff)
            size = np.asarray(temp_pcd.points, dtype=np.float32).shape[0]
        pcd1 = temp_pcd
        
        # for pcd2
        temp_pcd = pcd2.voxel_down_sample(voxel_size=coeff)
        size = np.asarray(temp_pcd.points, dtype=np.float32).shape[0]
        while size < 13000 or size > 14000:
            if size < 13000:
                coeff = coeff - 0.01
            else:
                coeff = coeff + 0.01
            temp_pcd = pcd2.voxel_down_sample(voxel_size=coeff)
            size = np.asarray(temp_pcd.points, dtype=np.float32).shape[0]
        pcd2 = temp_pcd
        
        # pcd1 = pcd1.voxel_down_sample(voxel_size=0.48)
        # pcd2 = pcd2.voxel_down_sample(voxel_size=0.48)
        
        # convert pointcloud to numpy array
        pcd1 = np.asarray(pcd1.points, dtype=np.float32)
        pcd2 = np.asarray(pcd2.points, dtype=np.float32)
        
        # normalize pointcloud
        pcd1 = pcd1 / self.range
        pcd2 = pcd2 / self.range
        
        
        # resize pointcloud to 13000,3
        pcd1 = pcd1[:13000,:]
        pcd2 = pcd2[:13000,:]
        pcd1 = np.transpose(pcd1, (1, 0))
        pcd2 = np.transpose(pcd2, (1, 0))
        
        # # concatenate pointcloud
        pcd = np.concatenate((pcd1, pcd2), axis=0)
        pcd = torch.from_numpy(pcd)
        
        pose1 = self.poses[idx]
        pose2 = self.poses[idx+1]
        
        # subtract label2 from label1
        pose = pose2 - pose1
        
        # drop last dimension
        pose = pose[:,0]
        pose = torch.from_numpy(pose)
        
        return pcd, pose
        
class KittiDataset(Dataset):
    def __init__(self, sequence="00", max_range = 120, path="/home/plnm/biloCNN/kitti"):
        self.sequence = sequence
        self.range = max_range
        self.path = path
        
        self.dataset = pykitti.odometry(self.path, self.sequence)
        self.poses = []
        
        self.homogenous_matrix = np.zeros((4, 4), dtype=np.float)
        self.rotation_matrix = np.zeros((3, 3), dtype=np.float)
        self.translation_matrix = np.zeros((3, 1), dtype=np.float)
        
        self.load_poses()
    
    def __len__(self):
        return len(self.dataset)
    
    def load_poses(self):
        
        for pose in self.dataset.poses:
            
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
            self.poses.append(np.concatenate((self.translation_matrix, euler_angles), axis=0))
    
    def __getitem__(self, index):
        if index >= len(self.dataset):
            raise IndexError("Index out of range")
        elif index < 0:
            raise IndexError("Index out of range")
        elif index == len(self.dataset) - 1:
            index = len(self.dataset) - 2
        
        # get consecutive frames of images from the dataset
        img1 = self.dataset.cam2_files(index)
        img2 = self.dataset.cam2_files(index+1)
        
        img1 = cv2.imread(img1)
        img2 = cv2.imread(img2)
        
        # concatenate images
        img = np.concatenate((img1, img2), axis=2)
        img = cv2.resize(img, (224, 224))
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32)
        img = img / 255.0
        img = torch.from_numpy(img)
        
        
        # get pointcloud data
        pcl1 = self.dataset.get_velo(index)[:,:3]
        pcl2 = self.dataset.get_velo(index+1)[:,:3]
        
        # convert pcd to pointcloud
        pcd1 = geometry.PointCloud()
        pcd1.points = utility.Vector3dVector(pcl1)
        
        pcd2 = geometry.PointCloud()
        pcd2.points = utility.Vector3dVector(pcl2)
        
        # compress the pointcloud to 13000-14000 points
        # for pcd1
        coeff = 0.4
        temp_pcd = pcd1.voxel_down_sample(voxel_size=coeff)
        size = np.asarray(temp_pcd.points, dtype=np.float32).shape[0]
        while size < 13000 or size > 14000:
            if size < 13000:
                coeff = coeff - 0.01
            else:
                coeff = coeff + 0.01
            temp_pcd = pcd1.voxel_down_sample(voxel_size=coeff)
            size = np.asarray(temp_pcd.points, dtype=np.float32).shape[0]
        pcd1 = temp_pcd
        
        # for pcd2
        temp_pcd = pcd2.voxel_down_sample(voxel_size=coeff)
        size = np.asarray(temp_pcd.points, dtype=np.float32).shape[0]
        while size < 13000 or size > 14000:
            if size < 13000:
                coeff = coeff - 0.01
            else:
                coeff = coeff + 0.01
            temp_pcd = pcd2.voxel_down_sample(voxel_size=coeff)
            size = np.asarray(temp_pcd.points, dtype=np.float32).shape[0]
        pcd2 = temp_pcd
        
        pcd1 = np.asarray(pcd1.points, dtype=np.float32)
        pcd2 = np.asarray(pcd2.points, dtype=np.float32)
        
        # normalize pointcloud
        pcd1 = pcd1 / self.range
        pcd2 = pcd2 / self.range
        
        # resize pointcloud to 13000,3
        pcd1 = pcd1[:13000,:]
        pcd2 = pcd2[:13000,:]
        pcd1 = np.transpose(pcd1, (1, 0))
        pcd2 = np.transpose(pcd2, (1, 0))
        
        pcd = np.concatenate((pcd1, pcd2), axis=0)
        pcd = torch.from_numpy(pcd)
        
        pose1 = self.poses[index]
        pose2 = self.poses[index+1]
        
        # subtract pose2 from pose1
        pose = pose2 - pose1
        
        # drop last dimension
        pose = pose[:,0]
        pose = torch.from_numpy(pose)
        
        return img, pcd, pose
        
        
if __name__ == "__main__":
    img = KittiIMGDataset()
    pcl = KittiPCLDataset()
    both = KittiDataset()
    
    
        
    
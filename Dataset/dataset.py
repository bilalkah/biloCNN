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



class KittiIMGAllDataset(Dataset):
    def __init__(self, sequence=["00"], path = "/home/plnm/biloCNN/kitti"):
        self.sequence = sequence
        self.path = path
        self.dataset = []
        
        for seq in self.sequence:
            self.dataset.append(pykitti.odometry(self.path, seq))
        
        self.poses = []
        
        self.load_poses()

    def load_poses(self):
        for k in range(len(self.sequence)):
            img_k = 0
            for pose in self.dataset[k].poses:
                homogenous_matrix = np.zeros((4, 4), dtype=np.float)
                
                for i in range(3):
                    for j in range(4):
                        homogenous_matrix[i][j] = pose[i][j]
                homogenous_matrix[3][3] = 1
                
                self.poses.append([homogenous_matrix,k,img_k])
                img_k += 1
                
        
    def __len__(self):
        return len(self.poses)-1

    def __getitem__(self, idx):
        if idx >= len(self.poses):
            print(idx)
            raise IndexError("Index out of range")
        elif idx < 0:
            print(idx)
            raise IndexError("Index out of range")
        
        dataset_k = self.poses[idx][1]
        img_k = self.poses[idx][2]
        
        if img_k == len(self.dataset[dataset_k])-1:
            idx -= 1
            dataset_k = self.poses[idx][1]
            img_k = self.poses[idx][2] 
        
        img1 = cv2.imread(self.dataset[dataset_k].cam2_files[img_k])
        img2 = cv2.imread(self.dataset[dataset_k].cam2_files[img_k+1])
        
        # concatenate images
        img = np.concatenate((img1, img2), axis=2)
        img = cv2.resize(img, (224, 224))
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32)
        img = img / 255.0
        img = torch.from_numpy(img)
        
        previous = self.poses[idx][0]
        current = self.poses[idx+1][0]
        
        rotation_prev = previous[:3, :3]
        translation_prev = previous[:3, 3]
        
        rotation_current = current[:3, :3]
        translation_current = current[:3, 3]
        
        rotation = np.dot(rotation_prev.T, rotation_current)
        translation = np.dot(rotation_prev.T, translation_current) - np.dot(rotation_prev.T, translation_prev)
        
        rotation_angles = cv2.Rodrigues(rotation)[0]
        # convert rotation angles shape from (3,1) to (3,)
        rotation_angles = rotation_angles.reshape(3)
        
        rotation_signs = np.ones(3, dtype=np.float)
        translation_signs = np.ones(3, dtype=np.float)
        
        for i in range(translation.shape[0]):
            if translation[i] < 0:
                translation_signs[i] = 0
                translation[i] = -translation[i]
            if rotation_angles[i] < 0:
                rotation_signs[i] = 0
                rotation_angles[i] = -rotation_angles[i]
        
        translation = np.concatenate((translation, translation_signs), axis=0)
        rotation = np.concatenate((rotation_angles, rotation_signs), axis=0)
        
        odometry = np.concatenate((translation, rotation), axis=0)
        
        # print(odometry.shape)
        
        # drop last dimension
        odometry = torch.from_numpy(odometry)
        odometry = odometry.type(torch.FloatTensor)
        return img, odometry




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
        
        self.load_poses()

    def load_poses(self):
        
        for pose in self.dataset.poses:
            homogenous_matrix = np.zeros((4, 4), dtype=np.float)
            for i in range(3):
                for j in range(4):
                    homogenous_matrix[i][j] = pose[i][j]
            
            homogenous_matrix[3][3] = 1
            
            self.poses.append(homogenous_matrix)
        
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
        
        previous = self.poses[idx]
        current = self.poses[idx+1]
        
        rotation_prev = previous[:3, :3]
        translation_prev = previous[:3, 3]
        
        rotation_current = current[:3, :3]
        translation_current = current[:3, 3]
        
        rotation = np.dot(rotation_prev.T, rotation_current)
        translation = np.dot(rotation_prev.T, translation_current) - np.dot(rotation_prev.T, translation_prev)
        
        rotation_angles = cv2.Rodrigues(rotation)[0]
        # convert rotation angles shape from (3,1) to (3,)
        rotation_angles = rotation_angles.reshape(3)
        
        rotation_signs = np.ones(3, dtype=np.float)
        translation_signs = np.ones(3, dtype=np.float)
        
        for i in range(translation.shape[0]):
            if translation[i] < 0:
                translation_signs[i] = 0
                translation[i] = -translation[i]
            if rotation_angles[i] < 0:
                rotation_signs[i] = 0
                rotation_angles[i] = -rotation_angles[i]
        
        translation = np.concatenate((translation, translation_signs), axis=0)
        rotation = np.concatenate((rotation_angles, rotation_signs), axis=0)
        
        odometry = np.concatenate((translation, rotation), axis=0)
        
        # print(odometry.shape)
        
        # drop last dimension
        odometry = torch.from_numpy(odometry)
        odometry = odometry.type(torch.FloatTensor)
        return img, odometry
        

class KittiPCLAllDataset(Dataset):
    def __init__(self, sequence=["00"], max_range = 120, path = "/home/plnm/biloCNN/kitti"):
        self.sequence = sequence
        self.path = path
        self.dataset = []
        self.range = max_range
        
        for seq in self.sequence:
            self.dataset.append(pykitti.odometry(self.path, seq))
        
        self.poses = []
        
        self.load_poses()

    def load_poses(self):
        for k in range(len(self.sequence)):
            pcl_k = 0
            for pose in self.dataset[k].poses:
                homogenous_matrix = np.zeros((4, 4), dtype=np.float)
                
                for i in range(3):
                    for j in range(4):
                        homogenous_matrix[i][j] = pose[i][j]
                homogenous_matrix[3][3] = 1
                
                self.poses.append([homogenous_matrix,k,pcl_k])
                pcl_k += 1
                
        
    def __len__(self):
        return len(self.poses)-1

    def __getitem__(self, idx):
        if idx >= len(self.poses):
            print(idx)
            raise IndexError("Index out of range")
        elif idx < 0:
            print(idx)
            raise IndexError("Index out of range")
        
        dataset_k = self.poses[idx][1]
        pcl_k = self.poses[idx][2]
        
        if pcl_k == len(self.dataset[dataset_k])-1:
            idx -= 1
            dataset_k = self.poses[idx][1]
            pcl_k = self.poses[idx][2] 
        
        # get pointcloud data
        pcl1 = self.dataset[dataset_k].get_velo(pcl_k)[:,:3]
        pcl2 = self.dataset[dataset_k].get_velo(pcl_k+1)[:,:3]
        
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
        while size < 13000 or size > 14500:
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
        while size < 13000 or size > 14500:
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
        
        previous = self.poses[idx][0]
        current = self.poses[idx+1][0]
        
        rotation_prev = previous[:3, :3]
        translation_prev = previous[:3, 3]
        
        rotation_current = current[:3, :3]
        translation_current = current[:3, 3]
        
        rotation = np.dot(rotation_prev.T, rotation_current)
        translation = np.dot(rotation_prev.T, translation_current) - np.dot(rotation_prev.T, translation_prev)
        
        rotation_angles = cv2.Rodrigues(rotation)[0]
        # convert rotation angles shape from (3,1) to (3,)
        rotation_angles = rotation_angles.reshape(3)
        
        rotation_signs = np.ones(3, dtype=np.float)
        translation_signs = np.ones(3, dtype=np.float)
        
        for i in range(translation.shape[0]):
            if translation[i] < 0:
                translation_signs[i] = 0
                translation[i] = -translation[i]
            if rotation_angles[i] < 0:
                rotation_signs[i] = 0
                rotation_angles[i] = -rotation_angles[i]
        
        translation = np.concatenate((translation, translation_signs), axis=0)
        rotation = np.concatenate((rotation_angles, rotation_signs), axis=0)
        
        odometry = np.concatenate((translation, rotation), axis=0)
        
        # print(odometry.shape)
        
        # drop last dimension
        odometry = torch.from_numpy(odometry)
        odometry = odometry.type(torch.FloatTensor)
        return pcd, odometry



# Dataset class for loading velodyne pointcloud data to tensor    
# convert pointcloud data to tensor
# data is stored in binary file
class KittiPCLDataset(Dataset):
    def __init__(self, sequence="00", max_range = 120, path = "/home/plnm/biloCNN/kitti"):
        self.sequence = sequence
        self.range = max_range
        self.path = path
        self.dataset = pykitti.odometry(self.path, self.sequence)

        self.poses = []
        
        self.load_poses()
    
    def load_poses(self):
        for pose in self.dataset.poses:
            homogenous_matrix = np.zeros((4, 4), dtype=np.float)
            for i in range(3):
                for j in range(4):
                    homogenous_matrix[i][j] = pose[i][j]
            
            homogenous_matrix[3][3] = 1
            
            self.poses.append(homogenous_matrix)
    
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
        while size < 13000 or size > 14500:
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
        while size < 13000 or size > 14500:
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
        
        previous = self.poses[index]
        current = self.poses[index+1]
        
        rotation_prev = previous[:3, :3]
        translation_prev = previous[:3, 3]
        
        rotation_current = current[:3, :3]
        translation_current = current[:3, 3]
        
        rotation = np.dot(rotation_prev.T, rotation_current)
        translation = np.dot(rotation_prev.T, translation_current) - np.dot(rotation_prev.T, translation_prev)
        
        rotation_angles = cv2.Rodrigues(rotation)[0]
        # convert rotation angles shape from (3,1) to (3,)
        rotation_angles = rotation_angles.reshape(3)
        
        # print(rotation_angles.shape)
        # print(translation, rotation_angles)
        rotation_signs = np.ones(3, dtype=np.float)
        translation_signs = np.ones(3, dtype=np.float)
        
        for i in range(translation.shape[0]):
            if translation[i] < 0:
                translation_signs[i] = 0
                translation[i] = -translation[i]
            if rotation_angles[i] < 0:
                rotation_signs[i] = 0
                rotation_angles[i] = -rotation_angles[i]
        
        translation = np.concatenate((translation, translation_signs), axis=0)
        rotation = np.concatenate((rotation_angles, rotation_signs), axis=0)
        
        odometry = np.concatenate((translation, rotation), axis=0)
        
        # print(odometry)
        # drop last dimension
        odometry = torch.from_numpy(odometry)
        return pcd, odometry
        
        
class KittiAllDataset(Dataset):
    def __init__(self, sequence=["00"], max_range = 120, path = "/home/plnm/biloCNN/kitti"):
        self.sequence = sequence
        self.path = path
        self.dataset = []
        self.range = max_range
        
        for seq in self.sequence:
            self.dataset.append(pykitti.odometry(self.path, seq))
        
        self.poses = []
        
        self.load_poses()

    def load_poses(self):
        for k in range(len(self.sequence)):
            data_k = 0
            for pose in self.dataset[k].poses:
                homogenous_matrix = np.zeros((4, 4), dtype=np.float)
                
                for i in range(3):
                    for j in range(4):
                        homogenous_matrix[i][j] = pose[i][j]
                homogenous_matrix[3][3] = 1
                
                self.poses.append([homogenous_matrix,k,data_k])
                data_k += 1
                
        
    def __len__(self):
        return len(self.poses)-1

    def __getitem__(self, idx):
        if idx >= len(self.poses):
            print(idx)
            raise IndexError("Index out of range")
        elif idx < 0:
            print(idx)
            raise IndexError("Index out of range")
        
        dataset_k = self.poses[idx][1]
        data_k = self.poses[idx][2]
        
        if data_k == len(self.dataset[dataset_k])-1:
            idx -= 1
            dataset_k = self.poses[idx][1]
            data_k = self.poses[idx][2] 
        
        img1 = cv2.imread(self.dataset[dataset_k].cam2_files[data_k])
        img2 = cv2.imread(self.dataset[dataset_k].cam2_files[data_k+1])
        
        # concatenate images
        img = np.concatenate((img1, img2), axis=2)
        img = cv2.resize(img, (224, 224))
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32)
        img = img / 255.0
        img = torch.from_numpy(img)
        
        
        
        # get pointcloud data
        pcl1 = self.dataset[dataset_k].get_velo(data_k)[:,:3]
        pcl2 = self.dataset[dataset_k].get_velo(data_k+1)[:,:3]
        
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
        while size < 13000 or size > 14500:
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
        while size < 13000 or size > 14500:
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
        
        previous = self.poses[idx][0]
        current = self.poses[idx+1][0]
        
        rotation_prev = previous[:3, :3]
        translation_prev = previous[:3, 3]
        
        rotation_current = current[:3, :3]
        translation_current = current[:3, 3]
        
        rotation = np.dot(rotation_prev.T, rotation_current)
        translation = np.dot(rotation_prev.T, translation_current) - np.dot(rotation_prev.T, translation_prev)
        
        rotation_angles = cv2.Rodrigues(rotation)[0]
        # convert rotation angles shape from (3,1) to (3,)
        rotation_angles = rotation_angles.reshape(3)
        
        # rotation_signs = np.ones(3, dtype=np.float)
        # translation_signs = np.ones(3, dtype=np.float)
        
        # for i in range(translation.shape[0]):
        #     if translation[i] < 0:
        #         translation_signs[i] = 0
        #         translation[i] = -translation[i]
        #     if rotation_angles[i] < 0:
        #         rotation_signs[i] = 0
        #         rotation_angles[i] = -rotation_angles[i]
        
        # translation = np.concatenate((translation, translation_signs), axis=0)
        # rotation = np.concatenate((rotation_angles, rotation_signs), axis=0)
        
        odometry = np.concatenate((translation, rotation_angles), axis=0)
        
        # print(odometry.shape)
        
        # drop last dimension
        odometry = torch.from_numpy(odometry)
        odometry = odometry.type(torch.FloatTensor)
        return img, pcd, odometry

class KittiDataset(Dataset):
    def __init__(self, sequence="00", max_range = 120, path="/home/plnm/biloCNN/kitti"):
        self.sequence = sequence
        self.range = max_range
        self.path = path
        
        self.dataset = pykitti.odometry(self.path, self.sequence)
        self.poses = []
        
        self.load_poses()
    
    def __len__(self):
        return len(self.dataset)
    
    def load_poses(self):
        
        for pose in self.dataset.poses:
            for pose in self.dataset.poses:
                homogenous_matrix = np.zeros((4, 4), dtype=np.float)
            for i in range(3):
                for j in range(4):
                    homogenous_matrix[i][j] = pose[i][j]
            
            homogenous_matrix[3][3] = 1
            
            self.poses.append(homogenous_matrix)
    
    def __getitem__(self, index):
        if index >= len(self.dataset):
            raise IndexError("Index out of range")
        elif index < 0:
            raise IndexError("Index out of range")
        elif index == len(self.dataset) - 1:
            index = len(self.dataset) - 2
        
        # get consecutive frames of images from the dataset
        img1 = self.dataset.cam2_files[index]
        img2 = self.dataset.cam2_files[index+1]
        
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
        while size < 13000 or size > 14500:
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
        while size < 13000 or size > 14500:
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
        
        previous = self.poses[index]
        current = self.poses[index+1]
        
        rotation_prev = previous[:3, :3]
        translation_prev = previous[:3, 3]
        
        rotation_current = current[:3, :3]
        translation_current = current[:3, 3]
        
        rotation = np.dot(rotation_prev.T, rotation_current)
        translation = np.dot(rotation_prev.T, translation_current) - np.dot(rotation_prev.T, translation_prev)
        
        rotation_angles = cv2.Rodrigues(rotation)[0]
        # convert rotation angles shape from (3,1) to (3,)
        rotation_angles = rotation_angles.reshape(3)
        
        # rotation_signs = np.ones(3, dtype=np.float)
        # translation_signs = np.ones(3, dtype=np.float)
        
        # for i in range(translation.shape[0]):
        #     if translation[i] < 0:
        #         translation_signs[i] = 0
        #         translation[i] = -translation[i]
        #     if rotation_angles[i] < 0:
        #         rotation_signs[i] = 0
        #         rotation_angles[i] = -rotation_angles[i]
        
        # translation = np.concatenate((translation, translation_signs), axis=0)
        # rotation = np.concatenate((rotation_angles, rotation_signs), axis=0)
        
        odometry = np.concatenate((translation, rotation_angles), axis=0)
        
        # print(odometry)
        # drop last dimension
        odometry = torch.from_numpy(odometry)
        return img, pcd, odometry
        
        
if __name__ == "__main__":
    img = KittiIMGDataset()
    pcl = KittiPCLDataset()
    both = KittiDataset()
    
    
        
    
import pykitti
import cv2
import numpy as np
import cv2
import sys
import time


base_path = "/home/bilal/Desktop/project/data_odometry_color/dataset/"
odom = pykitti.odometry(base_path, '00')

map_pose_matrix = odom.poses

initial_homogenous_matrix = np.zeros((4, 4), dtype=np.float)
# fill initial homogenous matrix diagonal with 1
for i in range(4):
    initial_homogenous_matrix[i][i] = 1


    


# calculate camera_link to world transformation
# using map_poses
for i in range(len(map_pose_matrix)-1):
    # get current frame
    current_frame = map_pose_matrix[i+1]
    # get previous frame
    previous_frame = map_pose_matrix[i]
    # extract rotation and translation matrix
    rotation_matrix = current_frame[0:3, 0:3]
    translation_matrix = current_frame[0:3, 3]
    # calculate rotation and translation between current and previous frame
    rotation_matrix_prev = previous_frame[0:3, 0:3]
    translation_matrix_prev = previous_frame[0:3, 3]
    # calculate rotation and translation between current and previous frame
    rotation = np.dot(rotation_matrix_prev.T, rotation_matrix)
    translation = np.dot(rotation_matrix_prev.T, translation_matrix) - np.dot(rotation_matrix_prev.T, translation_matrix_prev)
    
    # convert rotation and translation matrix to homogenous matrix
    
    current_homogenous_matrix = np.zeros((4, 4), dtype=np.float)
    # euler angles
    rotation_angles = cv2.Rodrigues(rotation)[0]
    
    # euler angles to rotation matrix
    rotation_matrix_euler = cv2.Rodrigues(rotation_angles)[0]
    
    
    # fill current homogenous matrix diagonal with rotation_matrix_euler and translation
    for i in range(3):
        for j in range(3):
            current_homogenous_matrix[i][j] = rotation_matrix_euler[i][j]
        current_homogenous_matrix[i][3] = translation[i]
    current_homogenous_matrix[3][3] = 1
    
    # print(current_homogenous_matrix)
    
    # calculate camera_link to world transformation
    initial_homogenous_matrix = np.dot(initial_homogenous_matrix, current_homogenous_matrix)
    
    # print translation in camera_link_to_world_transformation and 
    print(initial_homogenous_matrix[0:3, 3],translation_matrix)
    
    # sys.exit()
    # rot1 = cv2.Rodrigues(rotation_matrix)[0]
    # rot2 = cv2.Rodrigues(rotation_matrix_prev)[0]
    # rot = rot2 - rot1
    
    # previos pose + rotation + translation
    
    
    # trans = translation_matrix - translation_matrix_prev
    # rot = rotation_matrix - rotation_matrix_prev
    # # squared sum of trans
    # sum1 = np.sum(np.square(trans))
    
    # # squared sum of translation
    # sum2 = np.sum(np.square(translation))
    
    # print(translation,trans)
    # print(sum2,sum1)
    
    # sum3 = np.sum(np.square(rot))
    # sum4 = np.sum(np.square(rotation_angles))
    
    # print(rotation_angles,"\n", rot)
    # print(sum4,sum3)
    
    time.sleep(0.1)
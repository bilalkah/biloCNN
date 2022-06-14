import pykitti
import math 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import time
import numpy as np
file_path = "/home/bilal/Desktop/project/data_odometry_color/dataset/poses/01.txt"
out_file = "output.txt"
file = open(file_path,"r")
file_ = open(out_file,"r")
fig = plt.figure()
ax = fig.gca(projection='3d')
x=np.array([])
y=np.array([])
z=np.array([])

x_=np.array([])
y_=np.array([])
z_=np.array([])

data = []
for line in file:
    line = line.split(" ")
    line[-1] = line[-1].replace("\n","")
    data_x = float(line[3])
    data_y = float(line[7])
    data_z = float(line[11])
    x=np.append(x,data_x)
    y=np.append(y,data_y)
    z=np.append(z,data_z)

    data.append([data_x,data_y,data_z])
data_ = []
for line in file_:
    line = line.split(" ")
    line[-1] = line[-1].replace("\n","")
    data_x = float(line[3])
    data_y = float(line[7])
    data_z = float(line[11])
    data_.append([data_x,data_y,data_z])
    x_=np.append(x_,data_x)
    y_=np.append(y_,data_y)
    z_=np.append(z_,data_z)
    
print(len(data),len(data_))
# for i in range(data):
# # global referansa konum x,y,z'de
#     # read line from file and convert to float numpy array
#     # line = file.readline()
#     line = line.split(" ")
#     line[-1] = line[-1].replace("\n","")
#     data_x = float(line[3])
#     data_y = float(line[7])
#     data_z = float(line[11])
#     # print(data_x,data_y,data_z)
    
#     line_ = line_.split(" ")
#     line_[-1] = line_[-1].replace("\n","")    
#     data_x_ = float(line_[3])
#     data_y_ = float(line_[7])
#     data_z_ = float(line_[11])
    
#     #convert string to float
#     # line = np.array(line,dtype=np.float32)
    
    
#     x=np.append(x,data_x)
#     y=np.append(y,data_y)
#     z=np.append(z,data_z)

#     x_=np.append(x_,data_x_)
#     y_=np.append(y_,data_y_)
#     z_=np.append(z_,data_z_)

    
ax.scatter(x_, y_, z_, marker='o', color='red')
ax.scatter(x, y, z, marker='o', color='blue')
plt.show()
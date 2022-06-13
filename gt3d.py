import pykitti
import numpy
import math 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import time

fig = plt.figure()
ax = fig.gca(projection='3d')

x=numpy.array([])
y=numpy.array([])
z=numpy.array([])
roll=numpy.array([])
pitch=numpy.array([])
yaw=numpy.array([])

basedir = '/home/bilal/Desktop/project/data_odometry_color/dataset/'
sequence = '00'

dataset = pykitti.odometry(basedir, sequence)
for i in range(4541):
# global referansa konum x,y,z'de
    x=numpy.append(x,[dataset.poses[i][0][3]])
    y=numpy.append(y,[dataset.poses[i][1][3]])
    z=numpy.append(z,[dataset.poses[i][2][3]])

ax.scatter(x, y, z, marker='o')

plt.show()

v = numpy.array([0, 0, 1])
fig2 = plt.figure()
ax = fig2.gca(projection='3d')
ax.set_xlabel('X')
ax.set_zlabel('Z')
for i in range(3000,3020):
    r= R.from_matrix([[dataset.poses[i][0][0], dataset.poses[i][0][1], dataset.poses[i][0][2]], [dataset.poses[i][1][0], dataset.poses[i][1][1], dataset.poses[i][1][2]], [dataset.poses[i][2][0], dataset.poses[i][2][1], dataset.poses[i][2][2]]])
    v2=r.apply(v) # global referansa gore bakis acisi v2'de
    ax.quiver(x[i], y[i], z[i], v2[0], v2[1], v2[2])
plt.show()

fig3 = plt.figure()
ax = fig3.gca(projection='3d')
ax.set_xlabel('X')
ax.set_zlabel('Z')
for i in range(4000,4010):
    r1= R.from_matrix([[dataset.poses[i][0][0], dataset.poses[i][0][1], dataset.poses[i][0][2]], [dataset.poses[i][1][0], dataset.poses[i][1][1], dataset.poses[i][1][2]], [dataset.poses[i][2][0], dataset.poses[i][2][1], dataset.poses[i][2][2]]])
    r2= R.from_matrix([[dataset.poses[i+1][0][0], dataset.poses[i+1][0][1], dataset.poses[i+1][0][2]], [dataset.poses[i+1][1][0], dataset.poses[i+1][1][1], dataset.poses[i+1][1][2]], [dataset.poses[i+1][2][0], dataset.poses[i+1][2][1], dataset.poses[i+1][2][2]]])
    v1 = r1.apply(v)
    v2 = r2.apply(v)
    r3 = r2 * r1.inv()
    v3 = r3.as_rotvec()
    #print(r3.as_euler('xyz'))
    print(x[i], y[i], z[i], v3[0], v3[1], v3[2])
    ax.quiver(x[i], y[i], z[i], v1[0], v1[1], v1[2], color="red")
    ax.quiver(x[i], y[i], z[i], v3[0], v3[1], v3[2],length = 20)
    time.sleep(0.1)
plt.show()

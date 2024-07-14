# 1.旋转向量转旋转矩阵

import cv2
import numpy as np
 
 
def rotvector2rot(rotvector):
    Rm = cv2.Rodrigues(rotvector)[0]
    return Rm
 
 
rotvector = np.array([[0.223680285784755,	0.240347886848190,	0.176566110650535]])
print(rotvector2rot(rotvector))
 
# 输出
# [[ 0.95604131 -0.14593404  0.2543389 ]
#  [ 0.19907538  0.95986385 -0.19756111]
#  [-0.21529982  0.23950919  0.94672136]]



#  2.四元数转欧拉角
from scipy.spatial.transform import Rotation as R
 
def quaternion2euler(quaternion):
    r = R.from_quat(quaternion)
    euler = r.as_euler('xyz', degrees=True)
    return euler
 
quaternion = [0.03551,0.21960,-0.96928, 0.10494]
print(quaternion2euler(quaternion))
 
# 输出
# [ -24.90053735    6.599459   -169.1003646 ]


#  3.欧拉角转四元数
from scipy.spatial.transform import Rotation as R
 
 
def euler2quaternion(euler):
    r = R.from_euler('xyz', euler, degrees=True)
    quaternion = r.as_quat()
    return quaternion
 
 
euler = [-24.90053735, 6.599459, -169.1003646]
print(euler2quaternion(euler))
 
# 输出
# [ 0.03550998  0.21959986 -0.9692794   0.10493993]

# 5.旋转矩阵转欧拉角
import numpy as np
import math
 
 
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
 
def rot2euler(R):
    assert (isRotationMatrix(R))
 
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
 
    singular = sy < 1e-6
 
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2]) * 180 / np.pi
        y = math.atan2(-R[2, 0], sy) * 180 / np.pi
        z = math.atan2(R[1, 0], R[0, 0]) * 180 / np.pi
    else:
        x = math.atan2(-R[1, 2], R[1, 1]) * 180 / np.pi
        y = math.atan2(-R[2, 0], sy) * 180 / np.pi
        z = 0
 
    return np.array([x, y, z])
 
 
rot = np.array([[-1.01749712e-02,  9.99670705e-01, -2.35574076e-02],
 [-9.99890780e-01, -1.04241019e-02, -1.04769347e-02],
 [-1.07190495e-02,  2.34482322e-02,  9.99667586e-01]])
 
 
print(rot2euler(rot))
# 输出
# [  1.34368509   0.61416806 -90.58302646]
from scipy.spatial.transform import Rotation as R
 
 
def euler2rot(euler):
    r = R.from_euler('xyz', euler, degrees=True)
    rotation_matrix = r.as_matrix()
    return rotation_matrix
# euler = [0 , -0.785418, 0] #绕y轴逆时针90度
# euler = [0 , 0 , -0.785418] #绕z轴逆时针90度
euler = [-0.539519 * 180.0/3.1415926535,0.553419 * 180.0/3.1415926535,-0.518633 * 180.0/3.1415926535] # pku
# euler = [0.024383 , 0.003159 , 0.00002] #0.lua0
# euler = [1.595179, 0.003159, -0.785478] # 6t x +180 z -90
# euler = [0.024383, 1.573955, 0.785418]  # 6t y +180 z +90
# euler = [0.024383,    0.003159,   1.57081]  # 6t z90
# euler = [-1.546413,    0.003159,   0.00002]  # 6t x-90
# euler = [0.024383,    1.573955,   0.00002]  # 6t y90
print(euler2rot(euler))
 
 
# 输出
# [[-0.9754533   0.21902821 -0.02274859]
#  [-0.18783626 -0.88152702 -0.43316008]
#  [-0.11492777 -0.41825442  0.90102988]]

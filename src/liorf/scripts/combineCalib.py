import numpy as np
from scipy.spatial.transform import Rotation as R

# 假设两组 xyz rpy
xyz1 = np.array([1.0, 2.0, 3.0])
rpy1 = np.array([0.1, 0.2, 0.3])

xyz2 = np.array([0.5, -1.0, 2.5])
rpy2 = np.array([0.5, -0.3, 0.7])

# 转换为变换矩阵
T1 = R.from_euler('xyz', rpy1).as_matrix()
T1[:3, 3] = xyz1  # 不需要 reshape

T2 = R.from_euler('xyz', rpy2).as_matrix()
T2[:3, 3] = xyz2  # 不需要 reshape

# 相乘得到合并后的变换矩阵
T_combined = np.dot(T2, T1)

# 从合并后的变换矩阵提取合并后的 xyz rpy
xyz_combined = T_combined[:3, 3]
rpy_combined = R.from_matrix(T_combined[:3, :3]).as_euler('xyz')

print("合并后的 xyz:", xyz_combined)
print("合并后的 rpy:", rpy_combined)
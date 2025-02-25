import numpy as np
from scipy.optimize import minimize
import sys

# 假设的观测站位置和接收时间
# 每一行格式：[x, y, z, t]
data = np.array([
    [110.241, 27.204, 824, 100.767],
    [110.780, 27.456, 727, 112.220],
    [110.712, 27.785, 742, 188.020],
    [110.251, 27.825, 850, 258.985],
    [110.524, 27.617, 786, 118.443],
    [110.467, 27.921, 678, 266.871],
    [110.047, 27.121, 575, 163.024]
])

# 经纬度转换为米
data[:, 0] *= 97304  # 经度转换
data[:, 1] *= 111263  # 纬度转换
data[:, 2] += 200000  # 脱敏感处理，高程
data[:, 3] += 600     # 脱敏感处理，时间

# 速度c
c = 341  # 音速，单位：米/秒

# 初始猜测
guess = [np.mean(data[:,0]), np.mean(data[:,1]), np.mean(data[:,2]), np.min(data[:,3])]

def error_square_sum(x):
    ta, xa, ya, za = x
    sum_sq = 0
    residuals = []
    for row in data:
        xi, yi, zi, ti = row
        predicted_time = np.sqrt((xi - xa)**2 + (yi - ya)**2 + (zi - za)**2) / c
        residual = (ti - (ta + predicted_time))**2
        residuals.append(residual)
        sum_sq += residual
    return sum_sq, residuals

converged = False
while not converged:
    result = minimize(lambda x: error_square_sum(x)[0], guess, method='L-BFGS-B')
    _, residuals = error_square_sum(result.x)
    mask = np.array(residuals) > 600  # 找到大于500的残差对应的数据点
    if np.any(mask):
        data = data[~mask]  # 移除这些数据点
        print(f"Removed {np.sum(mask)} points, recalculating...")
    else:
        converged = True

ta, xa, ya, za = result.x

print(f"振动源时间: {ta} 秒")
print(f"振动源位置: ({xa}, {ya}, {za}) 米")
import numpy as np
from scipy.optimize import minimize
import  random
import sys
from scipy.stats import norm
# 假设的观测站位置和接收时间
# 每一行格式：[x, y, z, t]

#随机取出四个探测器
from itertools import combinations

random_numbers = [random.uniform(-0.5, 0.5) for _ in range(7)]


# 示例使用
#d = generate_random_numbers(7)
#d=[-0.4,-0.1,-0.4,0.1,0.4,0.5,-0.5]
d=random_numbers
global initial_guess
data = np.array([[-1.92939931e+04 ,-5.18326633e+04 , 8.37142857e+01 , 1.00767000e+02+d[0]],
 [ 3.34447749e+04, -2.37943873e+04 ,-1.32857143e+01 , 1.12220000e+02+d[1]],
 [ 3.14013909e+04 , 1.28111397e+04  ,1.71428571e+00 , 1.88020000e+02+d[2]],
 [-1.83209531e+04  ,3.95142597e+04  ,1.09714286e+02  ,2.58985000e+02+d[3]],
 [ 8.24303886e+03 ,-5.88104429e+03 , 4.57142857e+01  ,1.18443000e+02+d[4]],
 [ 2.69671086e+03 , 4.57449877e+04 ,-6.22857143e+01  ,2.66871000e+02+d[5]],
 [-3.81709691e+04 ,-1.65622923e+04 ,-1.65285714e+02  ,1.63024000e+02+d[6]],
                 [-71000, 1739, 200, 252.62 + 18 -0.2]           ]
)
h = [-70000,739,200,252.62+18-0.2]
s = []
'''
#脱敏感处理
data[:,2] =(data[:,2]) + 200000
data[:,3] =(data[:,3]) + 600
'''


# 音爆抵达时间保持不变
print("标准化后的数据（经度、纬度、高程、时间）:")
print(data)

# 速度c
c = 340  # 音速，单位：米/秒
# 定义误差平方和函数

#四个变量求初始值initial

# 已知数据
#使用四个探测器的数据，放到矩阵data0中
x = np.array(data[3:7,0])  # 接收器的x坐标
y = np.array(data[3:7,1])  # 接收器的y坐标
z = np.array(data[3:7,2])  # 接收器的z坐标
t = np.array(data[3:7,3])  # 接收到信号的时间
data0=np.array([x,y,z,t]).T
print(data0)
#以上取出了前四个探测器的数据
# 初始猜测（可以使用平均位置和最小时间作为起点）
guess = [np.mean(x), np.mean(y), np.mean(z), np.min(t)]
print("guess ",guess)

#带零的是用四个探测器解出初始解
def error_square_sum0(In0):
    if len(In0) != 4:
        raise ValueError("Input must contain exactly four elements.")
    xa0, ya0, za0, ta0 = In0
    sum_sq = 0
    # 确保data已正确定义并可用
    for row in data0:
        if len(row) != 4:
            continue  # 或者你可以选择抛出一个错误
        xi, yi, zi, ti = row
        predicted_time = np.sqrt((xi - xa0)**2 + (yi - ya0)**2 + (zi - za0)**2) / c
        sum_sq += (ti - (ta0 + predicted_time))**2
    return sum_sq
# 使用BFGS方法最小化误差平方和
# 显示优化过程

options = {'disp': False, 'maxiter': 100, 'gtol': 1e-6}

#bounds = [(-sys.float_info.max, sys.float_info.max)] * 4  # 变量的无界
result0 = minimize(error_square_sum0, guess, method='L-BFGS-B',  options=options).x

xa0, ya0, za0, ta0 = result0
initial_guess = result0
print("最初解是")
print(initial_guess)
print("时间残差平方和::",error_square_sum0(result0))
#已经解出初始解
def error_square_sum(x):
    if len(x) != 4:
        raise ValueError("Input must contain exactly four elements.")
    xa, ya, za, ta = x
    sum_sq = 0
    # 确保data已正确定义并可用
    for row in data:
        if len(row) != 4:
            continue  # 或者你可以选择抛出一个错误
        xi, yi, zi, ti = row
        predicted_time = np.sqrt((xi - xa)**2 + (yi - ya)**2 + (zi - za)**2) / c
        sum_sq += (ti - (ta + predicted_time))**2
    return sum_sq

# 初始猜测值
#initial_guess = [-2117.4527074405, 14844.590691705400, 646.789794535382,-200.8505764 ]
# 使用BFGS方法最小化误差平方和
# 显示优化过程
options = {'disp': True, 'maxiter': 100, 'gtol': 1e-6}

#bounds = [(-sys.float_info.max, sys.float_info.max)] * 4  # 变量的无界
result = minimize(error_square_sum, initial_guess, method='L-BFGS-B',  options=options)

xa, ya, za, ta = result.x
print(f"振动源时间: {ta} 秒")
print(f"振动源位置: ({xa}, {ya}, {za}) 米")
print("总时间残差平方和", error_square_sum(result.x))
print(d)
# 打印结果

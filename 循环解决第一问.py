import numpy as np
from scipy.optimize import minimize
import sys
# 假设的观测站位置和接收时间
# 每一行格式：[x, y, z, t]

#随机取出四个探测器
from itertools import combinations

def combination_generator(n):
    numbers = range(7)  # 生成数字0, 1, 2, 3, 4, 5, 6
    comb_gen = combinations(numbers, 4)  # 创建一个组合生成器
    for _ in range(n):
        try:
            yield next(comb_gen)  # 产生一个组合
        except StopIteration:  # 如果组合用尽，停止迭代
            break
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
data[:, 0] = data[:, 0] * 97304  # 经度转换
data[:, 1] = data[:, 1] * 111263  # 纬度转换
# 示例使用
for comb in combination_generator(30):  # 指定输出次数
    zuhe =comb
    print(zuhe)

    '''
    #脱敏感处理
    data[:,2] =(data[:,2]) + 200000
    data[:,3] =(data[:,3]) + 600
    '''

    # 速度c
    c = 341  # 音速，单位：米/秒
    # 定义误差平方和函数

    # 四个变量求初始值initial

    # 已知数据
    # 使用四个探测器的数据，放到矩阵data0中
    x = np.array(data[zuhe, 0])  # 接收器的x坐标
    y = np.array(data[zuhe, 1])  # 接收器的y坐标
    z = np.array(data[zuhe, 2])  # 接收器的z坐标
    t = np.array(data[zuhe, 3])  # 接收到信号的时间
    data0 = np.array([x, y, z, t]).T
    # 以上取出了前四个探测器的数据
    # 初始猜测（可以使用平均位置和最小时间作为起点）
    guess = [np.mean(x), np.mean(y), np.mean(z), np.min(t)]
    '''
    #from scipy.optimize import least_squares
    # 定义方程的残差函数
    def residuals(vector,):
        xa0, ya0, za0, ta0 = vector
        return t - (ta0 + np.sqrt((x - xa0)**2 + (y - ya0)**2 + (z - za0)**2) / c)

    # 初始猜测（可以使用平均位置和最小时间作为起点）
    guess = [np.mean(x), np.mean(y), np.mean(z), np.min(t)]
    #guess =[19600,-40000,50,0]
    # 进行最小二乘拟合
    result0 = least_squares(residuals, guess)
    #前面是用四个点先大概给出一个解，result0.x是输出的结果向量，赋值到initial_guess下面用作初始解，
    # 带0的量都是第一个阶段给出初始解的变量
    initial_guess = result0.x
    # 输出结果
    if result0.success:
        xa0, ya0, za0, ta0 = result0.x
        print(f"预估位置: (xa0={xa0}, ya0={ya0}, za0={za0})")
        print(f"预估时间: ta0={ta0}")
    else:
        print("Optimization did not converge.")
    '''


    # 带零的是用四个探测器解出初始解
    def error_square_sum0(x0):
        if len(x0) != 4:
            raise ValueError("Input must contain exactly four elements.")
        ta0, xa0, ya0, za0 = x0
        sum_sq = 0
        # 确保data已正确定义并可用
        for row in data0:
            if len(row) != 4:
                continue  # 或者你可以选择抛出一个错误
            xi, yi, zi, ti = row
            predicted_time = np.sqrt((xi - xa0) ** 2 + (yi - ya0) ** 2 + (zi - za0) ** 2) / c
            sum_sq += (ti - (ta0 + predicted_time)) ** 2
        return sum_sq


    # 使用BFGS方法最小化误差平方和
    # 显示优化过程
    options = {'disp': False, 'maxiter': 10, 'gtol': 1e-6}

    # bounds = [(-sys.float_info.max, sys.float_info.max)] * 4  # 变量的无界
    result0 = minimize(error_square_sum0, guess, method='L-BFGS-B', options=options)

    ta0, xa0, ya0, za0 = result0.x
    initial_guess = result0.x
    print("最初解是")
    print(initial_guess)


    # 已经解出初始解

    def error_square_sum(x):
        if len(x) != 4:
            raise ValueError("Input must contain exactly four elements.")
        ta, xa, ya, za = x
        sum_sq = 0
        # 确保data已正确定义并可用
        for row in data:
            if len(row) != 4:
                continue  # 或者你可以选择抛出一个错误
            xi, yi, zi, ti = row
            predicted_time = np.sqrt((xi - xa) ** 2 + (yi - ya) ** 2 + (zi - za) ** 2) / c
            sum_sq += (ti - (ta + predicted_time)) ** 2
        return sum_sq


    # 初始猜测值
    # initial_guess = [-2117.4527074405, 14844.590691705400, 646.789794535382,-200.8505764 ]
    # 使用BFGS方法最小化误差平方和
    # 显示优化过程
    options = {'disp': False, 'maxiter': 10000, 'gtol': 1e-6}

    # bounds = [(-sys.float_info.max, sys.float_info.max)] * 4  # 变量的无界
    result = minimize(error_square_sum, initial_guess, method='L-BFGS-B', options=options)

    ta, xa, ya, za = result.x
    print(f"振动源时间: {ta} 秒")
    print(f"振动源位置: ({xa}, {ya}, {za}) 米")
    # 打印结果


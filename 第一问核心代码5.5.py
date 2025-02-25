import numpy as np
from scipy.optimize import minimize
from functools import reduce
import sys
# 假设的观测站位置和接收时间
# 每一行格式：[x, y, z, t]下面是原数据
data = np.array([
    [110.241, 27.204, 824, 270.0],
    [110.780, 27.456, 727, 196.5],
    [110.712, 27.785, 742, 110.69],
    [110.251, 27.825, 850, 94.65],
    [110.524, 27.617, 786, 270.065],
    [110.467, 27.921, 678, 92.453],
    [110.047, 27.121, 575, 156.93]
])

#定义声速
c =341
# 经纬度转换为米
data[:, 0] = data[:, 0] * 97304  # 经度转换
data[:, 1] = data[:, 1] * 111263  # 纬度转换
'''
data = np.array([[7393.48,36835.772,10101.7,79.32],
                 [10243.566,29960,6668.8,73.46],
                 [4739.21,33475,389.53,64.66],
                 [9546.,34685,885,51],
                 [9779.09,33680,-8398,23.60],
                 [9527,34868,-389,47.36],
                 [11741,34680,1560,42.678]
                 ])

#脱敏感处理
data[:,2] =data[:,2]
data[:,3] =data[:,3]
'''
std = [0 for i in range(0, 4)]  # 初始化一个标准差列表，后面用得到
for i in range(4):
    std[i] = np.std(data[:, i])
    print("第" + str(i) + "组数据的标准差是" + str(std[i]))
    total_std = ((reduce(lambda x, y: x * y, std))*c)**0.25 #算一个总的标准差
    print(total_std)
    
# 标准化经度、纬度和高程,这里不放缩故不除标准差
for i in range(3):
    mean = np.mean(data[:, i])
    data[:, i] = (data[:, i] - mean)
    print("第"+str(i)+"组数据的平均值是"+str(mean))



# 音爆抵达时间保持不变
print("标准化后的数据（经度、纬度、高程、时间）:")
print(data)

# 速度c
c = 341  # 音速，单位：米/秒
# 定义误差平方和函数

#四个变量求初始值initial

# 已知数据
#使用四个探测器的数据，放到矩阵data0中
#选取的行在这里
chosen=[3,4,5,6]
x = np.array(data[chosen,0])  # 接收器的x坐标
y = np.array(data[chosen,1])  # 接收器的y坐标
z = np.array(data[chosen,2])  # 接收器的z坐标
t = np.array(data[chosen,3])  # 接收到信号的时间
data0=np.array([x,y,z,t]).T
#print("下面用来算初始解的四个值")
#print(data0)
#以上取出了前四个探测器的数据
# 初始猜测（可以使用平均位置和最小时间作为起点）
guess = [np.mean(x), np.mean(y), np.mean(z), np.min(t)]

#带零的是用四个探测器解出初始解
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
        predicted_time = np.sqrt((xi - xa0)**2 + (yi - ya0)**2 + (zi - za0)**2) / c
        sum_sq += (ti - (ta0 + predicted_time))**2
    return sum_sq
# 使用BFGS方法最小化误差平方和
# 显示优化过程
options = {'disp': False, 'maxiter': 12, 'gtol': 1e-6}

#bounds = [(-sys.float_info.max, sys.float_info.max)] * 4  # 变量的无界
result0 = minimize(error_square_sum0, guess, method='L-BFGS-B',  options=options)

ta0, xa0, ya0, za0 = result0.x
initial_guess = result0.x
print("最初解是")
print(initial_guess)
#已经解出初始解

#下面是优化迭代的函数
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
        predicted_time = (((xi - xa)**2 + (yi - ya)**2 + (zi - za)**2) / c**2)**0.5
        sum_sq += (ti - (ta + predicted_time))**2
    return sum_sq
# 上面使用BFGS方法最小化误差平方和
# options显示优化过程
options = {'disp': False, 'maxiter': 1000, 'gtol': 1e-6}
#会让算法爆掉的变量界限设置
#bounds = [(-sys.float_info.max, sys.float_info.max)] * 4  # 变量的无界
result = minimize(error_square_sum, initial_guess, method='L-BFGS-B',  options=options)
ta, xa, ya, za = result.x
#print(result)
#下面是检验残差不会过大的函数
#设置残差
epsilon = 500
def error_square_test(xa,ya,za,ta):
    # t是计数，记录到第几行 h记录是否进行优化，优化过则为1，否则为0
    t ,h = 0,0
    sum_sq = 0
    # 确保data已正确定义并可用
    for row in data:
        if len(row) != 4:
            continue  # 或者你可以选择抛出一个错误
        xi, yi, zi, ti = row
        #t是计数，记录到第几行 h记录是否进行优化，优化过则为1，否则为0
        t +=1
        predicted_time = (((xi - xa)**2 + (yi - ya)**2 + (zi - za)**2) / c**2)**0.5
        if  ((predicted_time > epsilon ) and h == 0):
            print("第"+t+"组方差大于"+str(epsilon)+"将要重新计算")
            data = np.delete(data, t, axis=0)
            h +=1
    return h
if  error_square_test(xa,ya,za,ta)> 0:
    result = minimize(error_square_sum, initial_guess, method='L-BFGS-B', options=options)
#打印一下目标函数的优化值
F =result.fun
#print(F)
ta, xa, ya, za = result.x
# 打印结果
print(f"振动源时间: {ta} 秒")
print(f"振动源位置: ({xa}, {ya}, {za}) 米")


#尝试给出评价标准，类比闵可夫斯基空间中的距离
a = 0.49
b = 1 - a**2
def distance(arr,xa,ya,za,ta):
    distance = ((arr[0] - xa)**2 + (arr[1] - ya)**2 + (arr[2] - za)**2 - (c*(arr[3] - ta)**2))
    return distance
def judgement1(xa,ya,za,ta):
    d = np.zeros((7,1))
    for i in range(0,7):
        d[i,0] = distance(data[i,:],xa,ya,za,ta)
        print("对第"+str(i)+"个点时空距离是"+str(d[i,0])),
    mean = np.mean(d)
    squareE = sum((d[i,0] - mean)**2 for i in range(0,7))/7#这个是平方的均值
    #评价函数表达式
    f = ((abs(b*squareE + (a-b)*mean**2))**0.5)/1000000
    return f
#输出评价值
judge = judgement1(xa,ya,za,ta)
print("评价指标的值是"+str(judge))
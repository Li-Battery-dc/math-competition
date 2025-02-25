import pyswarms as ps
import numpy as np
from scipy.optimize import minimize

data = np.array([
    [110.241, 27.204, 824, 164.229],
    [110.783, 27.456, 727, 169.362],
    [110.762, 27.785, 742, 156.9363],
    [110.251, 28.025, 850, 141.4094],
    [110.524, 27.617, 786, 86.216],
    [110.467, 28.081, 678, 175.482],
    [110.047, 27.521, 575, 103.738]
])
data[:,3]=data[:,3]
# 调用函数并打印结果


# 经纬度转换为米
data[:, 0] = data[:, 0] * 97304  # 经度转换
data[:, 1] = data[:, 1] * 111263  # 纬度转换
'''
for i in range(4):
    std[i] = np.std(data[:, i])
    print("第" + str(i) + "组数据的标准差是" + str(std[i]))
    total_std = ((reduce(lambda x, y: x * y, std))*c)**0.25 #算一个总的标准差
    print(total_std)
    '''

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
chosen=[3,4,5,2]
x = np.array(data[chosen,0])  # 接收器的x坐标
y = np.array(data[chosen,1])  # 接收器的y坐标
z = np.array(data[chosen,2])  # 接收器的z坐标
t = np.array(data[chosen,3])  # 接收到信号的时间
data0=np.array([x,y,z,t]).T
# 定义优化问题的目标函数
import numpy as np
#选取若干探测器喂给梯度优化
chosen1=[3,4,5,2,6,1,0]
x = np.array(data[chosen1,0])  # 接收器的x坐标
y = np.array(data[chosen1,1])  # 接收器的y坐标
z = np.array(data[chosen1,2])  # 接收器的z坐标
t = np.array(data[chosen1,3])  # 接收到信号的时间
data1=np.array([x,y,z,t]).T
def error_square_sum0_pso(X):
    # X 是一个 n_particles x 4 的数组，每行是一个粒子的四维坐标
    # data0 是已知的四个四维点，形状为 4 x 4
    n_particles = X.shape[0]
    result = np.zeros(n_particles)

    for i in range(n_particles):
        # 计算第 i 个粒子到 data0 中每个点的minkowksi距离
        sum_sq = 0
        for row in data:  #在这里更改丢给函数的数据！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            xi, yi, zi, ti = row
            predicted_time = np.sqrt((xi - X[i,0]) ** 2 + (yi - X[i,1]) ** 2 + (zi - X[i,2]) ** 2) / c
            sum_sq += (ti - (X[i,3] + predicted_time)) ** 2
        # 计算距离之和并存储在 result 数组中
        result[i] = np.sum(sum_sq)

    return result


# 设置优化器的参数
options = {
    'c1': 0.25,   # 认知（个体）加速系数
    'c2': 0.2,   # 社会（群体）加速系数
    'w': 1, # 惯性权重
}

# 创建一个 GlobalBestPSO 对象
optimizer = ps.single.GlobalBestPSO(
    n_particles=50,       # 粒子数量
    dimensions=4,         # 问题的维度
    options=options       # 优化器的参数
)

# 运行优化器进行优化，迭代10000次
cost, pos = optimizer.optimize(error_square_sum0_pso, iters=1500)

print("粒子群给出初始解：", pos)
print("时间残差：", cost)
initial_guess = [6158.703142859042, -26714.53242857149, 31.31428571428569,18.89]
def error_square_sum(x):
    if len(x) != 4:
        raise ValueError("Input must contain exactly four elements.")
    ta, xa, ya, za = x
    sum_sq = 0
    # 确保data已正确定义并可用
    for row in data:#这里修改投喂的数据!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if len(row) != 4:
            continue  # 或者你可以选择抛出一个错误
        xi, yi, zi, ti = row
        predicted_time = (((xi - xa)**2 + (yi - ya)**2 + (zi - za)**2) / c**2)**0.5
        sum_sq += (ti - (ta + predicted_time))**2
    return sum_sq
# 使用BFGS方法最小化误差平方和
# 显示优化过程
options_BFGS = {'disp': False, 'maxiter': 1, 'gtol': 1e-6}

#bounds = [(-sys.float_info.max, sys.float_info.max)] * 4  # 变量的无界
result = minimize(error_square_sum, initial_guess, method='L-BFGS-B',  options=options_BFGS)
#print(result)
#打印一下目标函数的优化值
F =result.fun
print(F)
ta, xa, ya, za = result.x

# 打印结果
print(f"振动源时间: {ta} 秒")
#print(f"振动源位置: ({xa+10745447.526857141}, {ya+3066551.3324285713}, {za+740.2857142857143} ")
print(f"振动源位置: ({xa}, {ya}, {za} ")
np.mean(data[:, i])

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
       # print("对第"+str(i)+"个点时空距离是"+str(d[i,0])),
    mean = np.mean(d)
    squareE = sum((d[i,0] - mean)**2 for i in range(0,7))/7#这个是平方的均值
    #评价函数表达式
    f = ((abs(b*squareE + (a-b)*mean**2))**0.5)/1000000
    return f
#输出评价值
judge = judgement1(xa,ya,za,ta)
print("评价指标的值是"+str(judge))

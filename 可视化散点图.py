import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


data = np.array([
    [110.241, 27.204, 824, 100.767],
    [110.780, 27.456, 727, 112.220],
    [110.712, 27.785, 742, 188.020],
    [110.251, 27.825, 850, 258.985],
    [110.524, 27.617, 786, 118.443],
    [110.467, 27.921, 678, 266.871],
    [110.047, 27.121, 575, 163.024]
])

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
print(data)
#选取的行在这里
chosen=[0,1,2,3,4,5,6]
x = np.array(data[chosen,0])  # 接收器的x坐标
y = np.array(data[chosen,1])  # 接收器的y坐标
z = np.array(data[chosen,2])  # 接收器的z坐标
t = np.array(data[chosen,3])  # 接收到信号的时间
data0=np.array([x,y,z,t]).T
# 生成数据

# 创建三维坐标系
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# 设置坐标轴范围
ax.set_xlim([-40000, 40000])  # 设置x轴范围
ax.set_ylim([-60000, 40000])  # 设置y轴范围
ax.set_zlim([-100,100 ])  # 设置z轴范围
# 绘制三维散点图
ax.scatter(x, y, z, color='b', marker='o')

# 设置坐标轴标签
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

# 显示图形
plt.show()
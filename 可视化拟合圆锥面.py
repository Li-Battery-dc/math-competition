import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 生成数据
x = np.random.rand(100)
y = np.random.rand(100)
z = np.random.rand(100)

# 创建三维坐标系
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制三维散点图
ax.scatter(x, y, z, color='b', marker='o')

# 生成圆锥面的数据
a = 340#半顶角
x0,y0,z0 = 1,1,1#顶角坐标
theta = np.linspace(0, 2*np.pi, 100)
r = np.linspace(0, 1, 100)
R, THETA = np.meshgrid(r, theta)
X = R * np.cos(THETA)+x0
Y = R * np.sin(THETA)+y0
Z = a*R+z0

# 绘制圆锥面
ax.plot_surface(X, Y, Z, alpha=0.5)

# 设置坐标轴标签
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

# 显示图形
plt.show()
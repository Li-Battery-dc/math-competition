'''import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
#选取的行在这里
x = np.array([20220.571357962162, 33748.711566625156, 18403.885963329954, 12882.573141893728, 26681.454419937425, 16829.225847213333, 19940.538686661617])
y = np.array([-44953.291808058035, -57609.25881855261, -46295.212147428414, -37443.66072717015, -56792.22943604032, -41180.47317420234, -47415.47467088821])
z = np.array([567.0339590822373, -346.5893451150367, 557.085059806648, 666.5555353757863, 582.7263155544186, 612.4215598135874, 351.53247054131225])
t = np.array([-4.771737831066223, -53.28695015175921, -4.479146584110873, 10.074017976875755, -32.41702249786904, 1.187699478890205, -11.767790994684315])
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
plt.show()'''
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 选取的行在这里
x = np.array([1.07516617e+7, 1.07520307e+7, 1.07327722e+7, 12882.573141893728])
y = np.array([-44953.291808058035, -57609.25881855261, -46295.212147428414, -37443.66072717015, -56792.22943604032, -41180.47317420234, -47415.47467088821])
z = np.array([567.0339590822373, -346.5893451150367, 557.085059806648, 666.5555353757863, 582.7263155544186, 612.4215598135874, 351.53247054131225])
t = np.array([-4.771737831066223, -53.28695015175921, -4.479146584110873, 10.074017976875755, -32.41702249786904, 1.187699478890205, -11.767790994684315])
for i in range(0,7):
    t[i] = 340*t[i]
# 创建三维坐标系
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

R = 1000
theta = np.linspace(0, 2 * np.pi, 100)
fai = np.linspace(0, np.pi, 100)
FAI, THETA = np.meshgrid(fai, theta)
X = R * np.sin(FAI) * np.cos(THETA) + x[0]
Y = R * np.sin(FAI) * np.sin(THETA) + y[0]
Z = R * np.cos(FAI) + z[0]
ax.plot_surface(X, Y, Z, alpha=0.3)
# 设置坐标轴标签
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
## 绘制三维散点图和点标签
for i in range(len(x)):
    ax.scatter(x[i], y[i], z[i], color='b', marker='o')
    ax.text(x[i], y[i], z[i], f'{i}', color='red', fontsize=12)


# 绘制散点图后添加一点代码
from matplotlib.widgets import Slider

plt.subplots_adjust(bottom=0.25)
#设置显示范围
ax.set_xlim([15000, 25000])  # 设置x轴范围
ax.set_ylim([-40000,-50000])  # 设置y轴范围
ax.set_zlim([-300, 600])  # 设置y轴范围
# 设置坐标轴刻度尺寸一致
#ax.axis('equal')
axcolor = 'lightgoldenrodyellow'
ax_x = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
ax_y = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_z = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)

s_x = Slider(ax_x, 'XZ Rotation', 0, 360, 0)
s_y = Slider(ax_y, 'Y Rotation', 0, 360, 0)
s_z = Slider(ax_z, 'T Rotation', 0, 360, 0)

def update(val):
    ax.view_init(elev=s_x.val, azim=s_y.val)
    ax.yaxis.label.set_rotation(s_z.val)
    fig.canvas.draw_idle()

s_x.on_changed(update)
s_y.on_changed(update)
s_z.on_changed(update)

# 显示图形
plt.show()
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 选取的行在这里
t1,t2,t3,t4,t5,t6,t7={270.065, 196.583, 110.696, 94.653, 86.216, 67.274,206.789}
data = np.array([
    [110.241, 27.204, 824, t1],
    [110.783, 27.456, 727, t2],
    [110.762, 27.785, 742, t3],
    [110.251, 27.025, 850, t4],
    [110.524, 27.617, 786, t5],
    [110.467, 28.081, 678, t6],
    [110.047, 27.521, 575, t7]
])
data[:, 0] = data[:, 0] * 97304  # 经度转换
data[:, 1] = data[:, 1] * 111263  # 纬度转换
x = data[:,0]
y = data[:,1]
z = data[:,2]
t = data[:,3]
x_z_together = np.array([(np.sqrt(x[i]**2 + z[i]**2)) for i in range(0,7)])

for i in range(0,7):
    t[i] = 340*t[i]
# 创建三维坐标系
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制三维散点图
#ax.scatter(x_z_together, y, 0, color='r', marker='o')
# 设置坐标轴标签
ax.set_xlabel('XZ-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('T-axis')
#点标签
for i in range(len(x)):
    ax.scatter(x_z_together[i], y[i], t[i], color='b', marker='o')
    ax.text(x_z_together[i], y[i], t[i], f'{i}', color='red', fontsize=12)

# 绘制散点图后添加一点代码
from matplotlib.widgets import Slider

plt.subplots_adjust(bottom=0.25)

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
'''
a = 340#半顶角
x0,y0,z0,t0= 1.07516617e+07,3.03970104e+06,7.16805402e+02,1.86226042e+01#顶角坐标
xz0=np.sqrt(x0**2+z**2)
theta = np.linspace(0, 2*np.pi, 100)
r = np.linspace(0, 1, 100)
R, THETA = np.meshgrid(r, theta)
XZ = R * np.cos(THETA)+xz0
Y = R * np.sin(THETA)+y0
T = a*R+t0

# 绘制圆锥面
ax.plot_surface(XZ, Y, T, alpha=0.5)
'''
# 显示图形
plt.show()
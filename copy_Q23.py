import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import minimize

# 声速常量
c = 341

# 模拟的监测器数据
data = np.array([
    [110.241, 27.204, 824, 100.767, 164.229, 214.850, 270.065],
    [110.783, 27.456, 727, 92.453, 112.220, 169.362, 196.583],
    [110.762, 27.785, 742, 75.560, 110.696, 156.936, 188.020],
    [110.251, 28.025, 850, 94.653, 141.409, 196.517, 258.985],
    [110.524, 27.617, 786, 78.600, 86.216, 118.443, 126.669],
    [110.467, 28.081, 678, 67.274, 166.270, 175.482, 266.871],
    [110.047, 27.521, 575, 103.738, 163.024, 206.789, 210.306]
])

# 数据预处理，转换监测器坐标
coords = data[:, 0:3]
times = data[:, 3:7]

# 将经纬度转换为大致的米
coords[:, 0] *= 97304 # 经度每度的米
coords[:, 1] *= 111263  # 纬度每度的米

# 第一步和第二步：选出标准机和验证机，求解时空坐标
standard_machines = coords[:4, :]
standard_times = times[:4, :]
validation_machines = coords[4:, :]
validation_times = times[4:, :]

# 第二步：枚举和求解
def error_square_sum0(In0):
    if len(In0) != 4:
        raise ValueError("Input must contain exactly four elements.")
    ta0, xa0, ya0, za0 = In0
    sum_sq = 0
    # 确保data已正确定义并可用
    for row in data0:
        if len(row) != 4:
            continue  # 或者你可以选择抛出一个错误
        xi, yi, zi, ti = row
        predicted_time = np.sqrt((xi - xa0)**2 + (yi - ya0)**2 + (zi - za0)**2) / c
        sum_sq += (ti - (ta0 + predicted_time))**2
    return sum_sq

standard_code_combination=np.zeros((256,4))
result_combination=np.zeros((256,4))
code=0
for idx1 in range(4):
    for idx2 in range(4):
        for idx3 in range(4):
            for idx4 in range(4):
                t1=standard_times[0,idx1]
                t2=standard_times[1,idx2]
                t3=standard_times[2,idx3]
                t4=standard_times[3,idx4]
                standard_code_combination[code,:]=[idx1,idx2,idx3,idx4]
                x = np.array(standard_machines[:,0])  # 接收器的x坐标
                y = np.array(standard_machines[:,1])  # 接收器的y坐标
                z = np.array(standard_machines[:,2])  # 接收器的z坐标
                t = [t1,t2,t3,t4]
                data0=np.array([x,y,z,t]).T
                # print(data0)
                initial_guess=[np.mean(x), np.mean(y), np.mean(z), np.min(t)]
                # 使用BFGS方法最小化误差平方和
                options = {'disp': False, 'maxiter': 10, 'gtol': 1e-6}
                #bounds = [(-sys.float_info.max, sys.float_info.max)] * 4  # 变量的无界
                result_combination[code,:] = minimize(error_square_sum0,initial_guess, method='L-BFGS-B',  options=options).x
                F =  minimize(error_square_sum0,initial_guess, method='L-BFGS-B',  options=options).fun
                print(F)
                code+=1
 # 第三步：验证和求解最优解
# 这里的验证过程需要详细计算和枚举验证机情况，具体实现略过
# 初始化误差和解向量               
errors=np.zeros((256,1))
validation_code_combination =np.zeros((256,3))
# 计算解和时空坐标的匹配程度
def residuals(params, coords, fact_times):
    x, y, z, t0 = params
    distances = np.sqrt((x - coords[:, 0])**2 + (y - coords[:, 1])**2 + (z - coords[:, 2])**2)
    theoretical_times = t0 + distances / c
    sum_time_residuals=0
    fact_times=np.array(fact_times)
    sum_time_residuals = np.sum(np.square(fact_times - theoretical_times))
    return sum_time_residuals

temp_code_combination=np.zeros((64,3))
temp_res=np.zeros((64,1))
for index in range(256):
    temp_code=0
    for idx1 in range(4):
        for idx2 in range(4):
            for idx3 in range(4):
                temp_code_combination[temp_code,:]=[idx1,idx2,idx3]
                t1=validation_times[0,idx1]
                t2=validation_times[1,idx2]
                t3=validation_times[2,idx3]
                temp_time_combination=[t1,t2,t3]
                temp_res[temp_code]=residuals(result_combination[index,:],validation_machines,temp_time_combination)
                temp_code+=1
    #接下来寻找res最小的组合，作为一个看似合理的解答
    res_min=1e+10
    res_min_idx=[]
    for idxx in range(64):
        if(temp_res[idxx]<=res_min):
            res_min=temp_res[idxx]
            res_min_idx=temp_code_combination[idxx,:]
    errors[index]=res_min
    if len(res_min_idx) > 0:#某次循环中可能为空列表
        validation_code_combination[index,:] = res_min_idx
    print("第",index,"种假设的验证时空距离为",res_min)
    print(result_combination[index])
    print("对应的四个时间坐标顺序为：",standard_code_combination[index])
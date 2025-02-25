import numpy as np
import random
from scipy.stats import norm
from scipy.optimize import minimize
# 假设的观测站位置和接收时间
# 每一行格式：[x, y, z, t]
global initial_guess
i_guess = np.array([
    [1.07516617e+07,3.03970104e+06,7.16805402e+02,1.86226042e+01],
    [1.07520307e+07,3.10871350e+06,7.14047928e+02,1.96481211e+01],
    [1.07327722e+07,3.07569538e+06,7.15728078e+02,1.97239789e+01],
    [1.07698315e+07,3.07718504e+06,7.46180350e+02,2.25939248e+01]
])
data0 = np.zeros((4,4))
#四个独立爆炸源的信息
# 'a': 0.0, 'b': 1.0, 'c': 3.0, 'd': 3.0, 'e': 2.0, 'f': 3.0, 'g': 1.0
data1 = np.array([
    [110.241, 27.204, 824, 100.767],
    [110.783, 27.456, 727, 112.220],
    [110.762, 27.785, 742, 188.020],
    [110.251, 28.025, 850, 258.985],
    [110.524, 27.617, 786, 118.443],
    [110.467, 28.081, 678, 266.871],
    [110.047, 27.521, 575, 163.024]
])
# 'a': 3.0, 'b': 3.0, 'c': 1.0, 'd': 0.0, 'e': 3.0, 'f': 0.0, 'g': 3.0
data2 = np.array([
    [110.241, 27.204, 824, 270.065],
    [110.783, 27.456, 727, 196.583],
    [110.762, 27.785, 742, 110.696],
    [110.251, 28.025, 850, 94.653],
    [110.524, 27.617, 786, 126.669],
    [110.467, 28.081, 678, 67.2746],
    [110.047, 27.521, 575, 210.306]
])
# 'a': 1.0, 'b': 2.0, 'c': 2.0, 'd': 1.0, 'e': 1.0, 'f': 2.0, 'g': 0.0
data3 = np.array([
    [110.241, 27.204, 824, 164.229],
    [110.783, 27.456, 727, 169.362],
    [110.762, 27.785, 742, 156.9363],
    [110.251, 28.025, 850, 141.4094],
    [110.524, 27.617, 786, 86.216],
    [110.467, 28.081, 678, 175.482],
    [110.047, 27.521, 575, 103.738]
])
#'a': 2.0, 'b': 0.0, 'c': 0.0, 'd': 2.0, 'e': 0.0, 'f': 1.0, 'g': 2.0
data4 = np.array([
    [110.241, 27.204, 824, 214.850],
    [110.783, 27.456, 727, 92.453],
    [110.762, 27.785, 742, 75.560],
    [110.251, 28.025, 850, 196.517],
    [110.524, 27.617, 786, 78.600],
    [110.467, 28.081, 678, 166.270],
    [110.047, 27.521, 575, 206.789]
])
print(i_guess)
for i in range(3):
    if i == 0 :
        mens_k =  97304
    if i == 1 :
        mens_k =  111263
    if i == 2 :
        mens_k = 1
    mean = np.mean(data1[:, i]* mens_k)
    std = np.std(data1[:, i])
    print(mean,std)
    i_guess[:, i] = (i_guess[:, i] - mean)
print(i_guess)
k = 50.0
error_max_j = np.zeros((4,7))#行为序号
error_square_sum = np.zeros((4,1))
result_datas = np.zeros((4,4))#行为序号
def generate_random_numbers(k):
    random_numbers = [random.uniform(-0.5, 0.5) for _ in range(int(k))]
    return random_numbers

#返回权值
def normal_pdf(x):
    global k
    mean = 0  # 均值
    std_dev = 0.5  # 标准差
    return norm.pdf(x, loc = mean, scale = std_dev)

def data1_caculation (data1) :
    global data0
    global error_max_j
    global error_square_sum
    global result_datas
    data1[:, 0] = data1[:, 0] * 97304  # 经度转换
    data1[:, 1] = data1[:, 1] * 111263  # 纬度转换
    '''
    #脱敏感处理
    data1[:,2] =(data1[:,2]) + 200000
    data1[:,3] =(data1[:,3]) + 600
    '''
    #标准化经度、纬度和高程
    for i in range(3):
        mean = np.mean(data1[:, i])
        std = np.std(data1[:, i])
        data1[:, i] = (data1[:, i] - mean)
        print("第"+str(i)+"组数据的平均值和标准差是"+str(mean),str(std))



    # 音爆抵达时间保持不变
    print("标准化后的数据（经度、纬度、高程、时间）:")
    print(data1)

    # 速度c
    c = 340  # 音速，单位：米/秒
    # 定义误差平方和函数

    #四个变量求初始值initial

    # 已知数据
    #使用四个探测器的数据，放到矩阵data10中
    x = np.array(data1[3:7,0])  # 接收器的x坐标
    y = np.array(data1[3:7,1])  # 接收器的y坐标
    z = np.array(data1[3:7,2])  # 接收器的z坐标
    t = np.array(data1[3:7,3])  # 接收到信号的时间
    data0=np.array([x,y,z,t]).T
    print(data0)
    #以上取出了前四个探测器的数据
    # 初始猜测（可以使用平均位置和最小时间作为起点）
    guess = [np.mean(x), np.mean(y), np.mean(z), np.max(t)]
    print("guess ",guess)

    #带零的是用四个探测器解出初始解
    def error_square_sum0(In0):
        if len(In0) != 4:
            raise ValueError("Input must contain exactly four elements.")
        xa0, ya0, za0, ta0 = In0
        sum_sq = 0
        # 确保data1已正确定义并可用
        for row in data0:
            if len(row) != 4:
                continue  # 或者你可以选择抛出一个错误
            xi, yi, zi, ti = row
            predicted_time = np.sqrt((xi - xa0)**2 + (yi - ya0)**2 + (zi - za0)**2) / c
            global k
            for index in range (int(k)) :
                a = generate_random_numbers(k)
                sum_sq += normal_pdf(a[index]) * (ti + a[index] - (ta0 + predicted_time))**2
            return sum_sq
    # 使用BFGS方法最小化误差平方和
    # 显示优化过程

    options = {'disp': False, 'maxiter': 10, 'gtol': 1e-12}

    #bounds = [(-sys.float_info.max, sys.float_info.max)] * 4  # 变量的无界
    result0 = minimize(error_square_sum0, guess, method='L-BFGS-B',  options=options).x

    xa0, ya0, za0, ta0 = result0
    initial_guess = result0
    print("最初解是")
    print(initial_guess)
    print("时间残差平方和::",error_square_sum0(result0))
    #已经解出初始解
    def data1_error_square_sum(x):
        if len(x) != 4:
            raise ValueError("Input must contain exactly four elements.")
        xa, ya, za, ta = x
        sum_sq = 0
        # 确保data1已正确定义并可用
        for row in data1:
            if len(row) != 4:
                continue  # 或者你可以选择抛出一个错误
            xi, yi, zi, ti = row
            predicted_time = np.sqrt((xi - xa)**2 + (yi - ya)**2 + (zi - za)**2) / c
            global k
            for index in range (int(k)) :
                a = generate_random_numbers(k)
                sum_sq += normal_pdf(a[index]) * (ti + a[index] - (ta + predicted_time))**2
        return sum_sq

    # 初始猜测值
    #initial_guess = [-2117.4527074405, 14844.590691705400, 646.789794535382,-200.8505764 ]
    # 使用BFGS方法最小化误差平方和
    # 显示优化过程
    options = {'disp': False, 'maxiter': 50, 'gtol': 1e-12}

    #bounds = [(-sys.float_info.max, sys.float_info.max)] * 4  # 变量的无界
    data1_result = minimize(data1_error_square_sum, guess, method='L-BFGS-B',  options=options)

    xa, ya, za, ta = data1_result.x
    print(f"振动源时间: {ta} 秒")
    print(f"振动源位置: ({xa}, {ya}, {za}) 米")
    print("总时间残差平方和", data1_error_square_sum(data1_result.x))
    def data1_error_test(x) :
        m = 0
        if len(x) != 4:
            raise ValueError("Input must contain exactly four elements.")
        xa, ya, za, ta = x
        sum_sq = 0
        times = 0
        # 确保data1已正确定义并可用
        for row in data1:
            if len(row) != 4:
                continue  # 或者你可以选择抛出一个错误
            xi, yi, zi, ti = row
            predicted_time = np.sqrt((xi - xa)**2 + (yi - ya)**2 + (zi - za)**2) / c
            global k
            error_combination = np.zeros((int(k),1))
            for index in range (int(k)) :
                a = generate_random_numbers(k)
                error_combination[index,0] = normal_pdf(a[index]) * (ti + a[index] - (ta + predicted_time))**2
                sum_sq += normal_pdf(a[index]) * (ti + a[index] - (ta + predicted_time))**2
            error_max_j [m, times] = np.max(error_combination)
            times += 1
        error_square_sum [m, 0] = sum_sq
        result_datas [m, :] = data1_result.x
        return sum_sq
    # 打印结果
    data1_error_test(data1_result.x)
def data2_caculation (data2) :
    global data0
    global error_max_j
    global error_square_sum
    global result_datas
    data2[:, 0] = data2[:, 0] * 97304  # 经度转换
    data2[:, 1] = data2[:, 1] * 111263  # 纬度转换
    '''
    #脱敏感处理
    data[:,2] =(data[:,2]) + 200000
    data[:,3] =(data[:,3]) + 600
    '''
    #标准化经度、纬度和高程
    for i in range(3):
        mean = np.mean(data2[:, i])
        std = np.std(data2[:, i])
        data2[:, i] = (data2[:, i] - mean)
        print("第"+str(i)+"组数据的平均值和标准差是"+str(mean),str(std))



    # 音爆抵达时间保持不变
    print("标准化后的数据（经度、纬度、高程、时间）:")
    print(data2)

    # 速度c
    c = 340  # 音速，单位：米/秒
    # 定义误差平方和函数

    #四个变量求初始值initial

    # 已知数据
    #使用四个探测器的数据，放到矩阵data0中
    x = np.array(data2[3:7,0])  # 接收器的x坐标
    y = np.array(data2[3:7,1])  # 接收器的y坐标
    z = np.array(data2[3:7,2])  # 接收器的z坐标
    t = np.array(data2[3:7,3])  # 接收到信号的时间
    data0 = np.array([x,y,z,t]).T
    print(data0)
    #以上取出了前四个探测器的数据
    # 初始猜测（可以使用平均位置和最小时间作为起点）
    guess = [np.mean(x), np.mean(y), np.mean(z), np.max(t)]
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
            global k
            for index in range (int(k)) :
                a = generate_random_numbers(k)
                sum_sq += normal_pdf(a[index]) * (ti + a[index] - (ta0 + predicted_time))**2
        return sum_sq
    # 使用BFGS方法最小化误差平方和
    # 显示优化过程

    options = {'disp': False, 'maxiter': 10, 'gtol': 1e-12}

    #bounds = [(-sys.float_info.max, sys.float_info.max)] * 4  # 变量的无界
    result0 = minimize(error_square_sum0, guess, method='L-BFGS-B',  options=options).x

    xa0, ya0, za0, ta0 = result0
    initial_guess = result0
    print("最初解是")
    print(initial_guess)
    print("时间残差平方和::",error_square_sum0(result0))
    #已经解出初始解
    
    def data2_error_square_sum(x):
        if len(x) != 4:
            raise ValueError("Input must contain exactly four elements.")
        xa, ya, za, ta = x
        sum_sq = 0
        # 确保data已正确定义并可用
        for row in data2:
            if len(row) != 4:
                continue  # 或者你可以选择抛出一个错误
            xi, yi, zi, ti = row
            predicted_time = np.sqrt((xi - xa)**2 + (yi - ya)**2 + (zi - za)**2) / c
            for index in range (int(k)) :
                a = generate_random_numbers(k)
                sum_sq += normal_pdf(a[index]) * (ti + a[index] - (ta + predicted_time))**2
        return sum_sq

    # 初始猜测值
    #initial_guess = [-2117.4527074405, 14844.590691705400, 646.789794535382,-200.8505764 ]
    # 使用BFGS方法最小化误差平方和
    # 显示优化过程
    options = {'disp': False, 'maxiter': 50, 'gtol': 1e-12}

    #bounds = [(-sys.float_info.max, sys.float_info.max)] * 4  # 变量的无界
    data2_result = minimize(data2_error_square_sum, guess, method='L-BFGS-B',  options=options)

    xa, ya, za, ta = data2_result.x
    print(f"振动源时间: {ta} 秒")
    print(f"振动源位置: ({xa}, {ya}, {za}) 米")
    print("总时间残差平方和", data2_error_square_sum(data2_result.x))
    # 打印结果
    def data2_error_test(x) :
        m = 1
        if len(x) != 4:
            raise ValueError("Input must contain exactly four elements.")
        xa, ya, za, ta = x
        sum_sq = 0
        times = 0
        # 确保data1已正确定义并可用
        for row in data1:
            if len(row) != 4:
                continue  # 或者你可以选择抛出一个错误
            xi, yi, zi, ti = row
            predicted_time = np.sqrt((xi - xa)**2 + (yi - ya)**2 + (zi - za)**2) / c
            global k
            error_combination = np.zeros((int(k),1))
            for index in range (int(k)) :
                a = generate_random_numbers(k)
                error_combination[index,0] = normal_pdf(a[index]) * (ti + a[index] - (ta + predicted_time))**2
                sum_sq += normal_pdf(a[index]) * (ti + a[index] - (ta + predicted_time))**2
            error_max_j [m, times] = np.max(error_combination)
            times += 1
        error_square_sum [m, 0] = sum_sq
        result_datas [m, :] = data2_result.x
        return sum_sq
    data2_error_test(data2_result.x)
def data3_caculation (data3) :
    global data0
    global error_max_j
    global error_square_sum
    global result_datas
    data3[:, 0] = data3[:, 0] * 97304  # 经度转换
    data3[:, 1] = data3[:, 1] * 111263  # 纬度转换
    '''
    #脱敏感处理
    data[:,2] =(data[:,2]) + 200000
    data[:,3] =(data[:,3]) + 600
    '''
    #标准化经度、纬度和高程
    for i in range(3):
        mean = np.mean(data3[:, i])
        std = np.std(data3[:, i])
        data3[:, i] = (data3[:, i] - mean)
        print("第"+str(i)+"组数据的平均值和标准差是"+str(mean),str(std))



    # 音爆抵达时间保持不变
    print("标准化后的数据（经度、纬度、高程、时间）:")
    print(data3)

    # 速度c
    c = 340  # 音速，单位：米/秒
    # 定义误差平方和函数

    #四个变量求初始值initial

    # 已知数据
    #使用四个探测器的数据，放到矩阵data0中
    x = np.array(data3[3:7,0])  # 接收器的x坐标
    y = np.array(data3[3:7,1])  # 接收器的y坐标
    z = np.array(data3[3:7,2])  # 接收器的z坐标
    t = np.array(data3[3:7,3])  # 接收到信号的时间
    data0 = np.array([x,y,z,t]).T
    print(data0)
    #以上取出了前四个探测器的数据
    # 初始猜测（可以使用平均位置和最小时间作为起点）
    guess = [np.mean(x), np.mean(y), np.mean(z), np.max(t)]
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
            for index in range (int(k)) :
                a = generate_random_numbers(k)
                sum_sq += normal_pdf(a[index]) * (ti + a[index] - (ta0 + predicted_time))**2
        return sum_sq
    # 使用BFGS方法最小化误差平方和
    # 显示优化过程

    options = {'disp': False, 'maxiter': 10, 'gtol': 1e-12}

    #bounds = [(-sys.float_info.max, sys.float_info.max)] * 4  # 变量的无界
    result0 = minimize(error_square_sum0, guess, method='L-BFGS-B',  options=options).x

    xa0, ya0, za0, ta0 = result0
    initial_guess = result0
    print("最初解是")
    print(initial_guess)
    print("时间残差平方和::",error_square_sum0(result0))
    #已经解出初始解
    def data3_error_square_sum(x):
        if len(x) != 4:
            raise ValueError("Input must contain exactly four elements.")
        xa, ya, za, ta = x
        sum_sq = 0
        # 确保data已正确定义并可用
        for row in data3:
            if len(row) != 4:
                continue  # 或者你可以选择抛出一个错误
            xi, yi, zi, ti = row
            predicted_time = np.sqrt((xi - xa)**2 + (yi - ya)**2 + (zi - za)**2) / c
            for index in range (int(k)) :
                a = generate_random_numbers(k)
                sum_sq += normal_pdf(a[index]) * (ti + a[index] - (ta + predicted_time))**2
        return sum_sq

    # 初始猜测值
    #initial_guess = [-2117.4527074405, 14844.590691705400, 646.789794535382,-200.8505764 ]
    # 使用BFGS方法最小化误差平方和
    # 显示优化过程
    options = {'disp': False, 'maxiter': 50, 'gtol': 1e-12}

    #bounds = [(-sys.float_info.max, sys.float_info.max)] * 4  # 变量的无界
    data3_result = minimize(data3_error_square_sum, guess, method='L-BFGS-B',  options=options)

    xa, ya, za, ta = data3_result.x
    print(f"振动源时间: {ta} 秒")
    print(f"振动源位置: ({xa}, {ya}, {za}) 米")
    print("总时间残差平方和", data3_error_square_sum(data3_result.x))
    # 打印结果
    def data3_error_test(x) :
        m = 2
        if len(x) != 4:
            raise ValueError("Input must contain exactly four elements.")
        xa, ya, za, ta = x
        sum_sq = 0
        times = 0
        # 确保data1已正确定义并可用
        for row in data1:
            if len(row) != 4:
                continue  # 或者你可以选择抛出一个错误
            xi, yi, zi, ti = row
            predicted_time = np.sqrt((xi - xa)**2 + (yi - ya)**2 + (zi - za)**2) / c
            global k
            error_combination = np.zeros((int(k),1))
            for index in range (int(k)) :
                a = generate_random_numbers(k)
                error_combination[index,0] = normal_pdf(a[index]) * (ti + a[index] - (ta + predicted_time))**2
                sum_sq += normal_pdf(a[index]) * (ti + a[index] - (ta0 + predicted_time))**2
            error_max_j [m, times] = np.max(error_combination)
            times += 1
        error_square_sum [m, 0] = sum_sq
        result_datas [m, :] = data3_result.x
        return sum_sq
    data3_error_test(data3_result.x)
def data4_caculation (data4) :
    global data0
    global error_max_j
    global error_square_sum
    global result_datas
    data4[:, 0] = data4[:, 0] * 97304  # 经度转换
    data4[:, 1] = data4[:, 1] * 111263  # 纬度转换
    '''
    #脱敏感处理
    data[:,2] =(data[:,2]) + 200000
    data[:,3] =(data[:,3]) + 600
    '''
    #标准化经度、纬度和高程
    for i in range(3):
        mean = np.mean(data4[:, i])
        std = np.std(data4[:, i])
        data4[:, i] = (data4[:, i] - mean)
        print("第"+str(i)+"组数据的平均值和标准差是"+str(mean),str(std))



    # 音爆抵达时间保持不变
    print("标准化后的数据（经度、纬度、高程、时间）:")
    print(data4)

    # 速度c
    c = 340  # 音速，单位：米/秒
    # 定义误差平方和函数

    #四个变量求初始值initial

    # 已知数据
    #使用四个探测器的数据，放到矩阵data0中
    x = np.array(data4[3:7,0])  # 接收器的x坐标
    y = np.array(data4[3:7,1])  # 接收器的y坐标
    z = np.array(data4[3:7,2])  # 接收器的z坐标
    t = np.array(data4[3:7,3])  # 接收到信号的时间
    data0=np.array([x,y,z,t]).T
    print(data0)
    #以上取出了前四个探测器的数据
    # 初始猜测（可以使用平均位置和最小时间作为起点）
    guess = [np.mean(x), np.mean(y), np.mean(z), np.max(t)]
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
            for index in range (int(k)) :
                a = generate_random_numbers(k)
                sum_sq += normal_pdf(a[index]) * (ti + a[index] - (ta0 + predicted_time))**2
        return sum_sq
    # 使用BFGS方法最小化误差平方和
    # 显示优化过程

    options = {'disp': False, 'maxiter': 10, 'gtol': 1e-12}

    #bounds = [(-sys.float_info.max, sys.float_info.max)] * 4  # 变量的无界
    result0 = minimize(error_square_sum0, guess, method='L-BFGS-B',  options=options).x

    xa0, ya0, za0, ta0 = result0
    initial_guess = result0
    print("最初解是")
    print(initial_guess)
    print("时间残差平方和::",error_square_sum0(result0))
    #已经解出初始解
    def data4_error_square_sum(x):
        if len(x) != 4:
            raise ValueError("Input must contain exactly four elements.")
        xa, ya, za, ta = x
        sum_sq = 0
        # 确保data已正确定义并可用
        for row in data4:
            if len(row) != 4:
                continue  # 或者你可以选择抛出一个错误
            xi, yi, zi, ti = row
            predicted_time = np.sqrt((xi - xa)**2 + (yi - ya)**2 + (zi - za)**2) / c
            for index in range (int(k)) :
                a = generate_random_numbers(k)
                sum_sq += normal_pdf(a[index]) * (ti + a[index] - (ta + predicted_time))**2
        return sum_sq

    # 初始猜测值
    #initial_guess = [-2117.4527074405, 14844.590691705400, 646.789794535382,-200.8505764 ]
    # 使用BFGS方法最小化误差平方和
    # 显示优化过程
    options = {'disp': False, 'maxiter': 50, 'gtol': 1e-12}

    #bounds = [(-sys.float_info.max, sys.float_info.max)] * 4  # 变量的无界
    data4_result = minimize(data4_error_square_sum, guess, method='L-BFGS-B',  options=options)

    xa, ya, za, ta = data4_result.x
    print(f"振动源时间: {ta} 秒")
    print(f"振动源位置: ({xa}, {ya}, {za}) 米")
    print("总时间残差平方和", data4_error_square_sum(data4_result.x))
    # 打印结果
    def data4_error_test(x) :
        m = 3
        if len(x) != 4:
            raise ValueError("Input must contain exactly four elements.")
        xa, ya, za, ta = x
        sum_sq = 0
        times = 0
        # 确保data1已正确定义并可用
        for row in data1:
            if len(row) != 4:
                continue  # 或者你可以选择抛出一个错误
            xi, yi, zi, ti = row
            predicted_time = np.sqrt((xi - xa)**2 + (yi - ya)**2 + (zi - za)**2) / c
            global k
            error_combination = np.zeros((int(k),1))
            for index in range (int(k)) :
                a = generate_random_numbers(k)
                error_combination[index,0] = normal_pdf(a[index]) * (ti + a[index] - (ta + predicted_time))**2
                sum_sq += normal_pdf(a[index]) * (ti + a[index] - (ta0 + predicted_time))**2
            error_max_j [m, times] = np.max(error_combination)
            times += 1
        error_square_sum [m, 0] = sum_sq
        result_datas [m, :] = data4_result.x
        return sum_sq
    data4_error_test(data4_result.x)
# 以上完成了各带权残差平方和以及一些其他数据的计算
data1_caculation(data1)
data2_caculation(data2)
data3_caculation(data3)
data4_caculation(data4)
print("error_max_j", error_max_j)
print("i_guess",i_guess)
print("result_datas", result_datas)

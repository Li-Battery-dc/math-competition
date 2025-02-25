import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import minimize

# 声速常量
c = 340

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
coords[:, 0] *= 97304  # 经度每度的米
coords[:, 1] *= 111263  # 纬度每度的米

code_combination = np.zeros((16384,7))
result_combination = np.zeros((16384,4))
error_combination = np.zeros((16384,1))

def residuals(params, coords, fact_times):
    x, y, z, t0 = params
    distances = np.sqrt(np.square(x - coords[:, 0]) + np.square(y - coords[:, 1]) + np.square(z - coords[:, 2]))
    theoretical_times = t0 + distances / c
    sum_time_residuals=0
    fact_times=np.array(fact_times)
    #if(temp_code<=255):
    #    print("第",index,"种假设算出的理论时间是",params)
    sum_time_residuals = np.sum(np.square(fact_times - theoretical_times))
    return sum_time_residuals

code = 0
combination_temporary = np.zeros((1,7))
#对每个组合进行求解
data0 = np.zeros((7,4))
def Iteration_solve (combination_temporary,code) :
    global result_combination
    global error_combination
    global code_combination
    t = np.zeros((7,))
    for i in range(7):
        t[i,] = times[i, int(combination_temporary[0, i])]
    code_combination[code,:]=combination_temporary
    x = np.array(coords[:,0])  # 接收器的x坐标
    y = np.array(coords[:,1])  # 接收器的y坐标
    z = np.array(coords[:,2])  # 接收器的z坐标
    #print(x.shape,y.shape,z.shape,t.shape)
    global data0
    data0 = np.array([x,y,z,t]).T
    initial_guess=[np.mean(x), np.mean(y), np.mean(z), np.min(t)]
    # 使用BFGS方法最小化误差平方和
    options = {'disp': False, 'maxiter': 20, 'gtol': 1e-6}
    #bounds = [(-sys.float_info.max, sys.float_info.max)] * 4  # 变量的无界
    result_combination[code,:] = minimize(error_square_sum, initial_guess, method='L-BFGS-B',  options = options).x
    error_combination[code,0] = error_square_sum( result_combination[code, :] )

def error_square_sum(In0):
    global data0
    global code
    if len(In0) != 4:
        raise ValueError("Input must contain exactly four elements.")
    xa0, ya0, za0, ta0 = In0
    sum_sq = 0
    # 确保data0已正确定义并可用
    for row in data0:
        if len(row) != 4:
            continue  # 或者你可以选择抛出一个错误
        xi, yi, zi, ti = row
        predicted_time = np.sqrt((xi - xa0)**2 + (yi - ya0)**2 + (zi - za0)**2) / c
        sum_sq += (ti - (ta0 + predicted_time))**2
    
    return sum_sq
#组合生成并进行计算#无返回值

def combination (floor) :
    global code
    for idx in range(4) :
        combination_temporary[0,floor] = idx
        if(floor != 6):
            combination(floor+1)
        else :
            Iteration_solve(combination_temporary,code)
            code += 1

combination(0)

#print(data0)
num_point = 0
nums = 0
error_limit = 1000
for index in range (16384):
    if(error_combination[index,0] <= error_limit):
        nums += 1
#检查并记录可疑点数据
re_time_combination = np.zeros((nums,1))
re_error_combination = np.zeros((nums,1))
re_code_combination = np.zeros((nums,7))
for index in range (16384):
    if(error_combination[index,0] <= error_limit):
        re_time_combination[num_point,0] = result_combination[index,3]
        re_error_combination[num_point,0] = error_combination[index,0] 
        re_code_combination[num_point,:] = code_combination[index,:]
        num_point += 1
        print("第",index,"种假设的时间残差平方和为",error_combination[index,0])
        print("第",index,"种假设的信号索引为",code_combination[index,:])
for index in range (16384):
    if(error_combination[index,0] <= error_limit):
        print(result_combination[index, 3])
print("可疑点总数 : ",num_point)


#生成搜索字典
matrix_data = np.hstack((re_time_combination,re_error_combination,re_code_combination))

data_points = [
    {
        'value': row[0],
        'weight': row[1],
        'a': row[2],
        'b': row[3],
        'c': row[4],
        'd': row[5],
        'e': row[6],
        'f': row[7],
        'g': row[8]
    }
    for row in matrix_data
]
def is_valid_solution(solution):
    """检查解是否满足索引值分布条件"""
    indices = {idx: set() for idx in 'abcdefg'}
    for point in solution:
        for idx in 'abcdefg':
            indices[idx].add(point[idx])
    return all(len(indices[key]) == 4 for key in indices)  # 确保每个索引在解中恰好出现一次

def backtrack_search(sorted_points, start_index=0, current_solution=[], current_weight=0, best_weight=float('inf'), best_solution=None):
    if len(current_solution) == 4:
        if is_valid_solution(current_solution) and current_weight < best_weight:
            best_weight = current_weight
            best_solution = current_solution[:]
        return best_weight, best_solution
    
    for i in range(start_index, len(sorted_points)):
        point = sorted_points[i]
        if current_solution:
            # 检查value差距
            if not all(abs(point['value'] - other['value']) <= 5 for other in current_solution):
                continue
        
        next_solution = current_solution + [point]
        next_weight = current_weight + point['weight']
        new_best_weight, new_best_solution = backtrack_search(sorted_points, i + 1, next_solution, next_weight, best_weight, best_solution)
        '''
        if(new_best_weight <= 785):
            best_weight, best_solution = best_weight, best_solution
        
        else :
            best_weight, best_solution = (new_best_weight, new_best_solution) if new_best_weight < best_weight else (best_weight, best_solution)
        '''
        best_weight, best_solution = (new_best_weight, new_best_solution) if new_best_weight < best_weight else (best_weight, best_solution)
    return best_weight, best_solution

# 预处理数据并启动搜索
sorted_points = sorted(data_points, key=lambda x: x['value'])
best_weight, best_solution = backtrack_search(sorted_points)

print("最优解:", best_solution)
print("最小weight之和:", best_weight)

        

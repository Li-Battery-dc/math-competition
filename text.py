import numpy as np
data=np.array([
    [-9.544,30.518,-18.497,33.003,-41.591,23.614,5.972],
    [-15.344,33.239,28.414,13.685,47.787,43.694,19.943]
])
data=data**2
print(np.sum(data[0,:])/7)
print(np.sum(data[1,:])/7)
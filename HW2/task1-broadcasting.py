import time
import numpy as np
import torch

x = np.random.rand(100,64)
t = np.random.rand(50,64)
d = np.zeros((x.shape[0],t.shape[0]))

# Comparing time measurements

# (a) two for-loops over i,j
time1 = time.time()
for i in range(x.shape[0]):
    for j in range(t.shape[0]):
        d[i,j] = np.sum((x[i]-t[j])**2)
time2 = time.time()
        
print("Two for-loops time taken (s): " + str(time2 - time1))
print(d)

# (b) numpy broadcasting
time1 = time.time()
# reshape X, T such that you can use broadcasting
d = (np.square(x[:,np.newaxis]-t[np.newaxis,:]).sum(axis=2))
time2 = time.time()

print("Numpy broadcasting time taken (s): " + str(time2 - time1))
print(d)

# (c) pytorch cpu
x_torch = torch.from_numpy(x)
t_torch = torch.from_numpy(t)
time1 = time.time()
d = ((torch.unsqueeze(x_torch,dim=1)-torch.unsqueeze(t_torch,dim=0))**2).sum(axis=2)
time2 = time.time()

print("PyTorch CPU time taken (s): " + str(time2 - time1))
print(d)
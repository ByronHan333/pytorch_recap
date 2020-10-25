import torch
import numpy as np


# 1.   安装anaconda,pycharm, CUDA+CuDNN（可选），虚拟环境，pytorch，并实现hello pytorch查看pytorch的版本
print("hello pytorch {}".format(torch.__version__))
print(torch.cuda.is_available())

# 2.   张量与矩阵、向量、标量的关系是怎么样的？
# 标量是0维，向量是1维，矩阵是2维，张量是3维或者多维

# 3.   Variable“赋予”张量什么功能？
# Variable是autograd的数据类型，用于封装tensor，用于自动求导

# 4.   采用torch.from_numpy创建张量，并打印查看ndarray和张量数据的地址；
a=np.array([1,2,3])
t = torch.from_numpy(a)
print("numpy array address: {}".format(a.__array_interface__['data'][0]))
print("tensor address: {}".format(t.data_ptr()))

# 5.   实现torch.normal()创建张量的四种模式。
print(torch.normal(1., torch.arange(1.,6.,dtype=torch.float)))
print(torch.normal(torch.arange(1.,6.,dtype=torch.float), torch.arange(1.,6.,dtype=torch.float)))
print(torch.normal(torch.arange(1.,6.,dtype=torch.float), 1))
print(torch.normal(1., 6., size=(4,)))
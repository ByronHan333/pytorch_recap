import torch
import matplotlib.pyplot as plt
import numpy as np
# 1.      调整线性回归模型停止条件以及y = 2*x + (5 + torch.randn(20, 1))中的斜率，训练一个线性回归模型；

torch.manual_seed(10)
lr = 0.001 # setup learning rate

# create dataset
x = torch.rand(2000, 1) * 10
y = 3*x + (5+torch.randn(2000,1))

# create parameters
w=torch.randn((1), requires_grad=True)
b=torch.zeros((1), requires_grad=True)

# iterate
for iteration in range(10000):

    # forward propagation
    wx=torch.mul(w,x)
    y_pred=torch.add(wx,b)

    # calculate MSE loss
    loss = (0.5*(y-y_pred)**2).mean()

    # backward propagation
    loss.backward()

    # update parameters
    b.data.sub_(lr * b.grad)
    w.data.sub_(lr * w.grad)

    # zero out grad
    b.grad.zero_()
    w.grad.zero_()

    print("iteration: {}, loss is: {}, w is: {}, b is {}: ".format(iteration, loss.data.numpy(), w.data.numpy(), b.data.numpy()))

    if loss.data.numpy() < 0.01:
        break

#print final result
print("\nfinal w is: {}, final b is {}: ".format(w.data.numpy(), b.data.numpy()))


# 2.      计算图的两个主要概念是什么？
# 节点和边，节点是数据，汝向量，矩阵，张量。边是运算，加减乘除卷积

# 3.      动态图与静态图的区别是什么？
# 动态图是pytorch，一遍搭建一遍运算，动态图是tensorflow先搭建再运算


# if iteration % 20 == 0:
#     plt.scatter(x.data.numpy(), y.data.numpy())
#     plt.scatter(x.data.numpy(), y_pred.data.numpy(), lw=5)
#     plt.text(2, 20, 'Loss=/.4f'%loss.data.numpy())
#     plt.xlim(1.5, 10)
#     plt.ylim(8,28)
#     plt.title("\niteration: {}, loss is: {}, w is: {}, b is {}: ".format(iteration, loss.data.numpy(), w.data.numpy(), b.data.numpy()))
#     plt.pause(0.5)
#
#     if loss.data.numpy() < 1:
#         break




# t = torch.ones((2,3))
# print("\nt: {} \nt_shape: {}".format(t, t.shape))
# t_stack_0 = torch.stack([t,t,t,t], dim=0)
# print("\nt_stack: {} \nt_stack_shape: {}".format(t_stack_0, t_stack_0.shape))
#
# t_stack_1 = torch.stack([t,t,t,t], dim=1)
# print("\nt_stack: {} \nt_stack_shape: {}".format(t_stack_1, t_stack_1.shape))
#
# t_stack_2 = torch.stack([t,t,t,t], dim=2)
# print("\nt_stack: {} \nt_stack_shape: {}".format(t_stack_2, t_stack_2.shape))

# t=torch.ones((2,5))
# print("\nt: {} \nt_shape: {}".format(t, t.shape))

# t.data.sub_
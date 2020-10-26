# 1.      逻辑回归模型为什么可以进行二分类？
# sigmoid function的output的区间是[0,1]，可以用来map到probablity上

# 2.      采用代码实现逻辑回归模型的训练，并尝试调整数据生成中的mean_value，将mean_value设置为更小的值，例如1，或者更大的值，例如5，会出现什么情况？
# mean value 指的是随机数据的平均值，
# 平均值如果太小+1和-1的数据会重合在一起，BCE会高，可能最后停止的threshold永远不会达到，需要的iteration更多
# 太大会重合的更快iteration是更少

# 再尝试仅调整bias，将bias调为更大或者负数，模型训练过程是怎么样的？
# bias在这里是数据离mean value的距离，bias过大会使得gradient过小，这样不管多少个iteration都没有进展

import matplotlib.pyplot as plt

# -*- coding: utf-8 -*-
# @Time    : 2021/2/22 9:54 下午
# @Author  : qiuqc@mail.ustc.edu.cn
# @FileName: test.py
# import USTC_lab.data
#
# a = USTC_lab.data.experience.Experience()
from USTC_lab.env import ImageEnv
# from USTC_lab import os
import time
import copy
import  torch.nn as nn
import torch.nn.functional as F
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        print(t)
        # (1)input layer
        t = t[0]

        # (2)hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (3)hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (4)hidden linear layer
        t = t.reshape(-1, 12 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)

        # (5)hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)

        # (6)output layer
        t = self.out(t)
        # t = F.softmax(t,dim=1)
        return





def run():
    for i in range(100):
        yield i
    for y in range(1000):
        yield y

def testnp():
    import numpy as np
    while True:
        a = np.random.random([2**18])
        start = time.time()
        for x in a:
            np.log(x)
        # b = np.log(a)
        print(time.time() - start)

def testnp2():
    import numpy as np
    import numpy as np
    a = np.array([[1, 0, 0],
                  [0.1, 0.6, 0.3],
                  [0.3, 0.3, 0.4]
                  ])
    # print(a.argmax(axis=1))
    def random_choice_prob_index(p, axis=1):
        r = np.expand_dims(np.random.rand(p.shape[1 - axis]), axis=axis)
        print(p.cumsum(axis=axis) > r)
        return (p.cumsum(axis=axis) > r).argmax(axis=axis)

    print(random_choice_prob_index(a))


if __name__ == "__main__":
    n1 = Network()
    n2 = copy.copy(n1)
    d = {}
    for i, v in n1.named_parameters():
        d[i] = v
    print("--------------------------")
    n2.load_state_dict(d)
    # for i, v in n2.named_parameters():
    #     print(v)
    # testnp2()
    #x = Network()
    #x(3,4,5)
    # import time
    # import numpy as np
    # import torch
    # while True:
    #     x = np.random.random([1024,4,84,84])
    #     s = time.time()
    #     y = torch.tensor(x, device='cuda')
    #     print(time.time() - s)
    #     time.sleep(10)

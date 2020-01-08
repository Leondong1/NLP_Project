# -*- coding: utf-8 -*-
"""
@Author   : Leon
@Contact  : wangdongjie1994@gmail.com
@Time     : 2020-01-05 19:53
@File     : pytorch基础模块使用.py
@Software : PyCharm
"""

from torch import nn
import torch
from torch import optim


class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        # self.linear = nn.Linear(1, 1)
        self.w = torch.tensor([0.7], dtype=torch.double,requires_grad=True)
        self.b = torch.tensor([0.6], dtype=torch.double, requires_grad=True)

    def forward(self, x):
        # out = self.linear(x)
        out = self.w * x + self.b
        return out


# optimizer = optim.Adam(LinearModel().parameters(), lr=1e-2)
loss_function = nn.MSELoss()


if __name__ == '__main__':
    linearModel = LinearModel()
    print(linearModel.parameters())   # 显示结果是一个生成器
    for i in linearModel.parameters():
        print(i.dim())
    # y_predict = linearModel(torch.Tensor([3]))   # 实际上会先调用 __call__ 方法然后调用 forward() 方法
    # print(y_predict)

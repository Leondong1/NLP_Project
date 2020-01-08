# -*- coding: utf-8 -*-
"""
@Author   : Leon
@Contact  : wangdongjie1994@gmail.com
@Time     : 2020-01-05 21:06
@File     : pytorch模块完成线性回归.py
@Software : PyCharm
"""

from torch import nn
import torch
from torch import optim
import matplotlib.pyplot as plt

# 将模型在GPU上运行代码的方式
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 准备数据
x = torch.randn([50, 1], requires_grad=True)
y = 4 * x + 0.7

x = x.to(device)
y = y.to(device)


class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        out = self.linear(x)
        return out


linearModel = LinearModel()
linearModel = linearModel.to(device)
optimizer = optim.Adam(linearModel.parameters(), lr=1e-2)
loss_function = nn.MSELoss()


if __name__ == '__main__':
    for i in range(5000):
        optimizer.zero_grad()
        y_predict = linearModel(x)
        loss = loss_function(y, y_predict)
        loss.backward(retain_graph=True)
        optimizer.step()
        if i % 500 == 0:
            print(i, loss)

    # 模型评估
    linearModel.eval()
    y_pre = linearModel(x)
    y_pre = y_pre.detach().numpy()
    plt.scatter(x.data.numpy(), y.data.numpy(), c='g')
    plt.plot(x.data.numpy(), y_pre, c='r')
    plt.show()



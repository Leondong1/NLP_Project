# -*- coding: utf-8 -*-
"""
@Author   : Leon
@Contact  : wangdongjie1994@gmail.com
@Time     : 2020-01-03 20:29
@File     : 线性回归手动实现.py
@Software : PyCharm
"""

"""
1. 准备数据
2. 计算预测值
3. 计算损失，把参数的梯度置为0，进行反向传播
4. 更新参数
"""

import torch
import matplotlib.pyplot as plt

x = torch.rand([50, 1], requires_grad=True)
y_true = 4 * x + 0.8

# 计算预测值，准备参数
w = torch.rand([], requires_grad=True)
b = torch.tensor([0], dtype=torch.float, requires_grad=True)


def loss_function(y_true, y_predict):
    loss_fc = (y_true - y_predict).pow(2).mean()
    # 将参数的梯度置为0
    [i.grad.data.zero_() for i in [w, b] if i.grad is not None]
    loss_fc.backward(retain_graph=True)
    return loss_fc.data


def optimize(learning_rate):
    w.data -= learning_rate * w.grad.data
    b.data -= learning_rate * b.grad.data


if __name__ == '__main__':
    for i in range(5000):
        y_predict = w * x + b
        loss = loss_function(y_true, y_predict)
        if i % 500 == 0:
            print(i, loss)
        optimize(0.01)

    predict = x * w + b

    # 绘制图像
    plt.scatter(x.data.numpy(), y_true.data.numpy(), c="g")
    plt.plot(x.data.numpy(), predict.data.numpy(), c='r')
    plt.show()

    # 打印参数
    print('w', w.item())
    print('b', b.item())
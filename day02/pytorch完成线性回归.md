梯度：是一个向量，导数+变化最快的方向(学习的前进方向)
  梯度下降：参数往梯度反方向去走，损失函数的值能够最快的变小




嵌套函数： 一层一层求导，各层的导数相乘
多个变量：针对每个单独的变量求偏导时，暂时把其他变量都当作常量


计算图

反向传播



 out = y_predict

 y_predict(W, X)
 loss(y, y_predict)

 dloss/dy_predict
损失函数是关于预测输出值的函数
 dy_predict/dw
 预测输出值是关于参数的函数


 y_predict = x * w

 x, y  样本的特征值和目标值都是已知常数



y = f(z)
z=g(x)
y1 - y0  = f'(z1) (z1 - z0)
y0:需要拟合成为的值
y1:现在输出的值

z1:根据现在的x的值x1的输出值

因为不想存储f'(z1)

计算出来 我根据y0 需要x取到合适的位置是z应该取的值z0


传递残差的过程  根据 y0  y1 的差距，计算当前输出z1 和 z0的差距

以后再根据z1 和 z0的差距去优化x



初始化所有的参数

前向计算
  根据模型中计算的方式和输入数据，得到输出结果

  常量： 样本的特征
  变量： 参数（需要更新优化的值）
    torch.tensor(requires_grad=True)


    w = torch.tensor(requires_grad=True)
    loss
    
    loss.backward()


    w.grad
    w.grad.zero_()



    w = w - w.grad * learning_rate

nn.Module :
  定义模型的结构
  __init__
    创建好所有需要用到的参数（tensor）
    创建网络模型的组件：全连接层/循环神经网络

   因为很多时候一个神经元里面有很多的参数，咱们则不太好自己去定义我们的每一个参数，使用网络模型的组件更加容易

  **forward(self, x):**

  从输入到输出的一次计算过程（一次前向计算）

​    返回预测输出(y_predict)

计算过程定义在forward 里面

损失函数
loss_func = nn.MSELoss()
loss = loss_func(y, y_predict)
优化
opt = torch.optim.SGD(参数, 学习率)
opt = torch.optim.Adam(参数, 学习率)

opt.zero_grad()
loss.backward()
opt.step()


nn.Linear(2,1).  这里阐述了为何2，1 为咱们特征的输入与输出，并且可以理解为w,b的值

(None, 2)  w: (2, 1)   (None, 1)
             b: (1,)

    x * w + b = y



在GPU上运行代码
  把模型的所有参数都转换成gpu类型的tensor
  把所有的数据也都转换成gpu类型的tensor

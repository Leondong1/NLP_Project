## pytorch中数据加载

batch:数据打乱顺序，组成一波一波的数据，批处理

epoch:那所有的数据训练一次

数据加载
  Dataset
    __init__(self):
    __getitem__(self, index):
      # 魔法方法 []
​    __len__(self):

      # 魔法方法 len()

  DataLoader
    随机获取批量数据
    参数：
      dataset: dataset 实例
      batch_size: 一批数据的数目
      shuffle: 是否打乱数据
      num_workers: 线程数


自带数据集
  torchvision.dataset.MNIST(root=, train=True, download=False, transform=)




图像，位图
每个像素长什么样子

单通道图片：每个像素只需要一个数值就能表示
三通道图片：每个像素需要3个数字才能表示


表示一张图片的tensor的形状
  0         1       2
  (height, width, channel)

  (channel, height, width)
  2          0       1

  .transpose().transpose()

  .permute()
torchvision.transforms.Compose([
torchvision.transforms.ToTensor()
torchvision.transforms.Normalize(mean, std)])


(None,1,28,28)

(None, 1 * 28 * 28)



(None, 10)   # 0/1/2/3/4/.../9



softmax(x0, x1, x2, x3, x4, x5, .., x9)
= (e^x0, e^x1, e^x2, e^x3, e^x4, e^x5, .., e^x9)/(e^x0 + e^x1 + e^x2 + e^x3, e^x4+ e^x5+ ..+ e^x9)


x0 = e^x0/(e^x0 + e^x1 + e^x2 + e^x3, e^x4+ e^x5+ ..+ e^x9)
x1 = e^x1/(e^x0 + e^x1 + e^x2 + e^x3, e^x4+ e^x5+ ..+ e^x9)

计算损失
对数似然的多分类版本：交叉熵

y_pre = [0.1, 0.1, 0.1, 0.05, ... 0.2]

y_true =[0,  0,  0,  0,  1, ......, 0]



交叉熵loss = - sum [y_true_i * log(y_predict_i)]

loss = - 1 * log(0.05)



(None, 10)   softmax

criterion = nn.CrossEntropyLoss()
loss = criterion(input,target)

#1. 对输出值计算softmax和取对数
output = F.log_softmax(x,dim=-1)
#2. 使用torch中带权损失
loss = F.nll_loss(output,target)
          0        1       2
样本     小猫    小狗    小刺猬
1         1        0       0
2         0        1      0
3         0        0      1


one-hot 编码


y_predict = w*x + b
loss = (y - y_predict)^2
    =  (y - w*x + b)^2
    = x^2*w^2  + A * b^2 +  B * wb


模型的保存和加载

y_predict
(None, 10)
(2, 10)
  0    1    2   3   4   5   6   7   8    9
[[0.1,0.1,0.1,0.19,0.1,0.1,0.1,0.1,0.1,0.01],
 [0.1,0.1,0.1,0.29,0.001,0.001,0.1,0.1,0.1,0.01]]

[3, 3]


y
(None, 10)
[[0,0,0,1,0,0,0,0,0,0],
 [0,0,0,0,0,0,1,0,0,0]]
[3, 6]


[3, 3]  ==   [3, 6]

[1, 0].mean()  0.5

[1,0,0].mean()   0.33

[1,1,1,1,1,1,1,1, 0,0].mean()  0.8




自然语言处理介绍
  tokenization
  分词
    现成解决方案：
      jieba
        # pip install jieba
        import jieba
        jieba.lcut('') # ['', '']



语言模型



P(“考研英语词汇”) = P('考'|start) * P('研'|'考') * P('英'|'考研') * P('语'|'考研英') * P('词'|'考研英语') * P('汇'|"考研英语词") * P(end|'考研英语词汇')


2,3
2-gram
P(“考研英语词汇”) = P('考'|start) * P('研'|'考') * P('英'|'研') * P('语'|'英') * P('词'|'语') * P('汇'|"词") * P(end|'汇')


P('考'|start) = '考'开头的文本数/所有字开头的文本数
P('研'|'考') = '考研'这个2字组合出现多少词/'考'字出现多少次




词/字  token   当作是一个类别   one-hot   UNK

5000个token


输入：5个token
输出：下一个token 是  5000个token


(None, 5, 5000)

(None, 5000)  # 5000个 token的概率


词语向量化（word2vec)
  非稀疏向量

skip-gram

_ _  ?   _ _

cbow

? ?  _  ? ?


青蛙
蟾蜍
的距离很小

男人  - 女人
king - queen
的距离相等


word embedding
  稠密矩阵

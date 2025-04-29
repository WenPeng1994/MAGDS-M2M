#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Auther:Evans_WP
@Contact:WenPeng19943251019@163.com
@Time:2024/2/29 21:05
@File:test.py
@Desc:****************
"""
import random


def Atest():
    import networkx as nx

    # 创建一个有向图
    G = nx.DiGraph()

    # 添加边
    G.add_edges_from([(1, 2), (2, 3), (2, 4), (3, 4)])

    # 获取节点 1 到节点 4 之间的所有简单路径
    all_paths = list(nx.all_simple_paths(G, source=1, target=4))

    # 打印所有路径
    for path in all_paths:
        print(path)

def Btest():
    print(all([1,1,0]))
    print(all([1,2,4]))
    a,b = [1,2]
    print(a)
    print(b)
    c = [int(i) for i in '1-2'.split('-')]
    print(c)


def Ctest():
    import numpy as np
    A = np.matrix([[1,2,3],[4,5,6],[7,8,9]])
    print(A[0:2,])
    print(A.shape)

def Dtest():
    A = {}
    # A['a'] = 2
    A.setdefault('a',dict())
    A['a'].setdefault('aa',1)
    print(A)

def Etest():
    import matplotlib.pyplot as plt
    a = [(1,2),(3,4)]
    a1 = [str(i) for i in a]
    b = [24,32]
    plt.barh(a1,b)
    plt.show()
    print([str((1,2))])

def Ftest():  # 循环测试
    num = 0
    while num < -3:
        print("数据小于3")
        num += 1
    else:
        print("数据等于3")

def Gtest():
    a = [1,1,2]
    a1 = [a]
    print(a1[1])

def Htest():
    import networkx as nx
    import matplotlib.pyplot as plt


    # 创建一个简单的图
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4])

    # 绘制图形，根据节点属性设置节点形状
    nx.draw(G, node_shape=['None','None','d','d'])

    # 显示图形
    plt.show()

def Itest():
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(0, 10)
    y = x ** 2

    markers = ['d', 'd', 'None']
    lines = ['None', '-', '-']
    colors = ['red', 'green', 'blue']

    for i in range(len(markers)):
        # I've offset the second and third lines so you can see the differences
        plt.plot(x, y + i * 10, marker=markers[i], linestyle=lines[i], color=colors[i])

    plt.show()

# 熵的变化测试
def Jtest():
    import numpy as np
    def vector_entropy(vector):
        probabilities = vector / np.sum(vector)  # 将向量归一化为概率分布
        entropy = -np.sum(probabilities * np.log2(probabilities))  # 计算熵
        return entropy,probabilities

    # 示例向量
    for i in range(1,6):
        vector = np.array([1,2,3,4,5])+i
        mean = np.mean(vector)
        var = np.var(vector)
        # 计算向量的熵
        entropy,probabilities = vector_entropy(vector)
        probabilities_mean = np.mean(probabilities)
        probabilities_var = np.var(probabilities)

        print(f"向量{vector}的均值为：{mean},方差为：{var},熵为:{entropy}\n"
              f"相应的概率为：{probabilities},均值为{probabilities_mean},方差为：{probabilities_var}")

        # 结论，熵能描述的仅仅是概率值的离散程度
        # 对于一般的数据来说，方差更能描述离散程度


def Ktest():
    import networkx as nx
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings("ignore", message="An issue occurred while importing 'torch-sparse'.*")
    warnings.filterwarnings("ignore", message="An issue occurred while importing 'pyg-lib'. *")
    from torch_geometric.utils.convert import to_networkx, from_networkx

    # 创建一个空的图
    G = nx.Graph()

    # 添加节点，并为节点添加属性
    G.add_node("s1", package=4)
    G.add_node("s2", package=5)
    G.add_node("s3", package=3)
    G.add_node("s4", package=6)

    # 添加边
    G.add_edge("s1","s2")
    G.add_edge("s2","s3")
    G.add_edge("s2","s4")

    # 绘制图形
    pos = nx.spring_layout(G)  # 定义节点位置
    nx.draw(G, pos, with_labels=True, node_size=1500, node_color="lightblue", font_size=12, font_weight="bold")

    # 绘制节点属性
    node_attrs = nx.get_node_attributes(G, 'package')
    for node, attr in node_attrs.items():
        x, y = pos[node]
        plt.text(x + 0.07, y + 0.07, attr, fontsize=13, color="red", verticalalignment="center",
                 horizontalalignment="left")
    plt.title("Simple Network Topology")
    plt.show()

    data = from_networkx(G)
    print(data)

    net_data = to_networkx(data)
    print(net_data)

def Ltest():
    dict0 = {}
    dict0[0] = 1
    print(dict0)
    dict0.setdefault(1,dict())  # 为字典的键设置初值，不会覆盖已有的键，可以用于设置数据类型，方便后续操作。
    print(dict0)

def Mtest():
    import networkx as nx

    # 创建一个图
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (1, 2)])

    # 获取节点0的邻居节点编号
    node = 0
    neighbors = list(G.neighbors(node))
    print("Node", node, "neighbors:", neighbors)


def Ntest():
    import networkx as nx
    G = nx.read_gpickle("graph.pkl")
    print(G.nodes[1].values())
    import networkx as nx

    # 创建一个图并添加节点和属性
    G = nx.Graph()
    G.add_node(0, color='red', weight=0.5)
    G.add_node(1, color='blue', weight=0.8)
    G.add_node(2, color='green', weight=0.3)

    # 获取所有节点的属性
    node_attributes = dict(G.nodes(data=True))

    # 打印所有节点的属性
    print("Node attributes:")
    for node, attributes in node_attributes.items():
        print("Node", node, ":", attributes)

def Otest():
    import networkx as nx
    import numpy as np

    # 创建一个图并添加节点和属性
    G = nx.Graph()
    G.add_node(0, color='red', weight=0.5)
    G.add_node(1, color='blue', weight=0.8)
    G.add_node(2, color='green', weight=0.3)

    # 获取某个属性的所有值

    colors = nx.get_node_attributes(G, 'color').values()
    weights = nx.get_node_attributes(G, 'weight').values()

    # 打印属性的所有值
    print("Colors:", list(colors))
    print("Weights:", np.array(list(weights)))
    x = np.concatenate(([list(colors)],[list(weights)]),axis=0)
    print(x)
    print(x.shape)
    node_attributes = dict(G.nodes(data=True))
    print(list(node_attributes[0].keys()))


def Ptest():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    from torch_geometric.datasets import Planetoid
    import torch.optim as optim

    # 定义 GCN 模型
    class GCN(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super(GCN, self).__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index

            # 第一层 GCNConv
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

            # 第二层 GCNConv
            x = self.conv2(x, edge_index)
            print(x)

            return F.log_softmax(x, dim=1)

    # 加载数据集
    dataset = Planetoid(root='../data/Cora', name='Cora')

    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(dataset.num_features, 16, dataset.num_classes).to(device)

    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()

    # 训练模型
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(dataset[0].to(device))
        loss = criterion(out[dataset[0].train_mask], dataset[0].y[dataset[0].train_mask])
        loss.backward()
        optimizer.step()

    # 测试模型
    model.eval()
    correct = 0
    total = 0
    for data in dataset:
        with torch.no_grad():
            out = model(data.to(device))
            pred = out.argmax(dim=1)
            correct += pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
            total += data.test_mask.sum().item()
    accuracy = correct / total
    print('Test Accuracy: {:.4f}'.format(accuracy))

def Qtest():
    import torch

    x = torch.tensor(1.0, requires_grad=True)

    # 条件逻辑
    if not x > 0:
        y = x ** 2
    else:
        y = x ** 3

    # 反向传播
    y.backward()

    # 打印梯度
    print("Gradient of x:", x.grad)

def Rtest():
    import torch
    a = torch.Tensor([[1],[0],[0]])
    print(a==1)
    vector = a.view(-1).tolist()

    print(vector==1)
    print(vector)


def Stest():
    import torch
    import numpy as np
    num = 5
    index = np.zeros(num,dtype=bool)
    index[[2,4]] = True
    print(index)

    x = torch.tensor([1.0]*5, requires_grad=True)

    print(x[index])
    y = torch.sum(x[index]**2)

    x_index = x[index]
    for i,ind in enumerate(index):
        if not ind:
            x_index = torch.insert(x_index,i,0)

    print('x_index',x_index)


    print(y)

    # 反向传播
    y.backward()

    # 打印梯度
    print("Gradient of x:", x.grad)

def Ttest():
    import torch

    # 创建一个张量
    x = torch.tensor([[1, 2, 3],
                      [4, 5, 6]])

    # 对张量进行填充
    padded_x = torch.nn.functional.pad(x, (0, 2), value=0)

    # 定义一个简单的网络
    model = torch.nn.Linear(5, 1)

    # 假设有一个目标
    target = torch.tensor([[0.1]])

    # 计算预测
    output = model(padded_x.view(1, -1))

    # 计算损失
    loss = torch.nn.functional.mse_loss(output, target)

    # 执行反向传播
    loss.backward()

    # 输出梯度
    print("Gradients of the model parameters:")
    print(model.weight.grad)

def Utest():
    import torch
    import torch.nn as nn

    class TextClassificationModel(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
            super(TextClassificationModel, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, num_classes)

        def forward(self, x):
            embedded = self.embedding(x)
            output, _ = self.rnn(embedded)
            output = self.fc(output[:, -1, :])  # 取最后一个时间步的输出
            return output

    # 定义模型参数
    vocab_size = 10000  # 假设词汇表大小为10000
    embedding_dim = 100
    hidden_dim = 128
    num_classes = 10

    # 创建模型实例
    model = TextClassificationModel(vocab_size, embedding_dim, hidden_dim, num_classes)

    # 定义输入序列（假设包含三个单词）
    input_indices = torch.tensor([1, 3, 5])  # 假设这是输入序列中每个单词的索引

    # 前向传播
    output = model(input_indices.unsqueeze(0))  # 在批次维度上增加一个维度

    print("Model Output:")
    print(output)

def Vtest():
    import torch

    # 创建一个一维向量
    vector = torch.tensor([1., 2., 3., 4., 5.],requires_grad=True)

    # 在索引为2的位置插入0
    index_to_insert = 2
    value_to_insert = 0

    # 创建一个与原始向量相同大小的零向量
    inserted_vector = torch.zeros(vector.size(0) + 1, dtype=vector.dtype)

    # 将原始向量的前半部分拼接到插入位置之前
    inserted_vector[:index_to_insert] = vector[:index_to_insert]

    # 将插入值插入到指定位置
    inserted_vector[index_to_insert] = value_to_insert

    # 将原始向量的后半部分拼接到插入位置之后
    inserted_vector[index_to_insert + 1:] = vector[index_to_insert:]

    print(inserted_vector)

    y = torch.sum(inserted_vector)
    y.backward()
    print(vector.grad)

def Wtest():
    import torch
    a = torch.tensor(list(range(5)),requires_grad=True,dtype=float)
    print(a)
    index = torch.tensor([1,0,1,0,0,1,0,1,0,1],dtype=bool)
    b = torch.zeros(len(index),dtype=a.dtype)
    b[index] = a
    print(b)
    print(b[index])

    y = torch.sum(b*2)
    print(y)
    y.backward()

    print(a.grad)


def Xtest():
    import numpy as np
    import torch
    print(np.where(np.array([1,3,4,2])==2))
    print(torch.zeros([3,3]))

def Ytest():
    import random

    # 一个示例列表
    numbers = [1, 2, 3, 4, 5]

    # 从列表中随机选择一个数
    selected_number = random.choice(numbers)
    print(numbers)
    print("随机选择的数:", selected_number)


def Ztest():
    import torch
    import numpy as np
    a = torch.tensor([1,2,3])
    b = np.array([0,1,0])
    print(a[b==1])

import pickle
class Data:
    def __init__(self, value):
        self.value = value
        self.x = 100

    # def __getstate__(self):
    #     return self.value
    #
    # def __setstate__(self, state):
    #     self.value = state



def AAtest():
    # 创建一个自定义类的实例
    data_instance = Data(42)

    # 将实例保存到文件
    with open('data.pkl', 'wb') as f:
        pickle.dump(data_instance, f)


def ABtest():
    import torch
    a = torch.tensor([1.,2.],requires_grad=True)

    b = torch.tensor(a)

    print(b.requires_grad)

    print(2*b)
    y = torch.sum(2*b)

    y.backward()

    print(a.grad)
    print(b.grad)

def ACtest():
    import logging
    import time
    # 定义日志输出格式
    # log_format = "%(asctime)s - %(levelname)s - %(message)s"
    # date_format = "%m/%d/%Y %H%M%S %p"
    # logging.basicConfig(filename="my.log",level=logging.DEBUG,format=log_format,datefmt=date_format)
    # logging.debug("This is a debug log.")
    formatted_time = time.strftime("%Y-%m-%d", time.localtime())
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%m/%d/%Y %H:%M:%S %p"
    logging.basicConfig(filename='./logging/my_{}.log'.format(formatted_time), level=logging.INFO, format=log_format,
                        datefmt=date_format)
    a = 1
    logging.info(f"This is a debug log{a}.")


def ADtest():
    import time
    local_time = time.localtime()
    print("本地结构化时间:", local_time)

    utc_time = time.gmtime()
    print("UTC结构化时间:", utc_time)

    formatted_time = time.strftime("%Y-%m-%d", local_time)
    print("格式化的本地时间:", formatted_time)

def AEtest():
    import logging

    # 创建logger
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)  # 设置日志记录级别

    # 创建一个handler，用于将日志写入文件
    file_handler = logging.FileHandler('my_app.log')
    file_handler.setLevel(logging.DEBUG)

    # 创建一个handler，用于输出日志到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # 创建一个formatter，并指定时间格式，例如: '年-月-日 时:分:秒'
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    # 设置formatter
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加handler到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # 日志信息
    logger.debug('这条信息将出现在控制台和文件中，并带有自定义时间格式。')
    logger.info('这条信息也将出现在控制台和文件中，并带有自定义时间格式。')
    logger.warning('这条警告信息也是，并带有自定义时间格式。')
    logger.error('这条错误信息同样如此，并带有自定义时间格式。')
    logger.critical('这条严重错误信息也不例外，并带有自定义时间格式。')

def AFtest():
    import torch
    a1 = torch.randn(2,4,15)
    a2 = torch.randn(4,4)
    result = torch.matmul(a2,a1)
    print(result)
    print(result.size())


def AGtest():
    # 导入数据集的包
    import torchvision.datasets
    # 导入dataloader的包
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter

    # 创建测试数据集
    test_dataset = torchvision.datasets.CIFAR10(root="./CIRFA10", train=False,
                                                transform=torchvision.transforms.ToTensor())
    # 创建一个dataloader,设置批大小为4，每一个epoch重新洗牌，不进行多进程读取机制，不舍弃不能被整除的批次
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

    # 测试数据集中第一张图片对象
    img, target = test_dataset[0]
    print(img.shape, target)

    # 打印数据集中图片数量
    print(len(test_dataset))

    # loader中对象
    for data in test_dataloader:
        imgs, targets = data
        print(imgs.shape)
        print(targets)

    # dataloader中对象个数
    print(len(test_dataloader))


def AHtest():
    import torch_geometric
    print(torch_geometric.__version__)

def AItest():
    for i in range(100):
        print(f"这是测试函数{i+100}")

def AJtest():
    a = [0,0,1,0,0,1,0]
    positions = [index for index,value in enumerate(a) if value==1]
    random_index, = random.sample(positions,1)
    lst = [0]*len(a)
    lst[random_index] = 1
    print(lst)


def AKtest():
    import numpy as np
    rows = 2**15
    cols = 15

    matrix = np.zeros((rows,cols))
    print(matrix)

def ALtest():
    import torch
    a = torch.tensor(list(range(5)), requires_grad=True, dtype=float)
    index = [0,2,4]
    one_matrix = torch.eye(5)[:,index]
    b = a.float() @ one_matrix.float()
    y = torch.sum(b * 1)
    # print(y)
    y.backward()
    print(a.grad)

def AMtest():
    x = 1 if 5<2 else 0
    print(x)

def ANtest():
    a = (1,2,3,4)
    b = (*a[0:3],5)
    print(b)

def AOtest():
    a = [(1,3),(2,3),(4,5)]
    b = [list(ai) for ai in a]
    bT = [tuple(b[j][i] for j in range(len(b))) for i in range(len(b[0]))]
    print(bT)

def APtest():
    n = 10
    for i in range(n):
        if i == n-1:
            print('到达终点')

def AQtest():
    num = 300
    reward = 2000
    unit = reward**(1/num)
    print(unit**num)

def ARtest():
    print(2**3/2)

def AStest():
    from scipy.io import savemat
    import math
    n = 100
    x = list(range(n))
    y = [[xi*2,xi**2,math.sin(xi)] for xi in x]
    data_dict = {f'{n}_x':x,f'{n}_y':y}
    savemat('my_data_dict.mat',data_dict)

def ATtest():
    for episode in range(0,10,2):
        print(episode)

# 经验池保存测试
def AOtest():
    import pickle
    from collections import deque

    class ReplayBuffer:
        def __init__(self, capacity):
            self.buffer = deque(maxlen=capacity)

        def add(self, experience):
            self.buffer.append(experience)

        def save(self, filename):
            with open(filename, 'wb') as f:
                pickle.dump(self.buffer, f)

        def load(self, filename):
            with open(filename, 'rb') as f:
                self.buffer = pickle.load(f)

        def __len__(self):
            return len(self.buffer)


    # 创建一个容量为1000的经验池
    replay_buffer = ReplayBuffer(1000)

    # 添加一些经验
    for i in range(20):
        state = i
        action = i % 3
        reward = i * 10
        next_state = i + 1
        done = False
        experience = (state, action, reward, next_state, done)
        replay_buffer.add(experience)

    # 保存经验池到文件
    replay_buffer.save('replay_buffer.pkl')

    # 创建一个新的经验池并加载数据
    new_replay_buffer = ReplayBuffer(1000)
    new_replay_buffer.load('replay_buffer.pkl')

    # 打印加载后的经验池大小
    print("Loaded buffer size:", len(new_replay_buffer))

def APtest():
    a = 'main_v130-100-buffer'
    if a.startswith('main_v'):
        print('这个函数正常')
        a_list = a.split('-')
        print(int(a_list[1])>0)
        b = 5 if 5>30 else 1
        print(b)

def AQtest():
    import textwrap
    from comments.bugtest import test_variable
    code = """
        a=[1,2]
        print(a[2])
    """
    test_variable(code,'a')

if __name__ == '__main__':
    AQtest()

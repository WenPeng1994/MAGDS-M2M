#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Auther:Evans_WP
@Contact:WenPeng19943251019@163.com
@Time:2024/2/25 19:39
@File:env_sim.py
@Desc:****************
"""
# 环境中包含三种对象，
# 第一种是产生数据的摄像头（Camera),其功能是产生数据，并将数据传输出去。
# 第二种是传递数据的交换机(Switch),其功能是转发数据，将接收到的数据转发给下一个设备。
# 第三种是收集数据的数据中心(Data_center),其功能是接收数据和存储数据。
import collections

class Camera:
    # 产生数据，指定数据量
    def __init__(self,capacity):
        """
        生成数据包
        :param num: 数据包数量
        """
        self.buffer = collections.deque(maxlen=capacity)


    def add_dataset(self,src,std,num):
        """
        添加需要处理的数据集
        :param src: 源节点
        :param std: 目的节点
        :param num: 数据包数量
        :return:
        """
        for order in range(1,num+1):
            self.buffer.append((src,std,order))


    def trans(self):
        """
        将数据传输出去
        :return:
        """
        trans_data = self.buffer.pop()
        return trans_data

    # 返回缓冲区中数据包数量
    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)


class Switch:
    # 形似一个缓冲区，数据结构上类似一个栈
    def __init__(self,capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列，先进先出

    # 添加数据
    def add(self,src,std,order): # 数据包包括源终点和第几个数据包
        self.buffer.append((src,std,order))

    # 传输数据
    def trans(self):
        """
        将数据传输出去
        :return:
        """
        trans_data = self.buffer.pop()
        return trans_data

    # 返回缓冲区中数据包数量
    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)



class Data_center:
    # 仅仅收集数据
    # 形似一个缓冲区，数据结构上类似一个栈
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列，先进先出

    # 添加数据
    def add(self, src, std, order):  # 数据包包括源终点和第几个数据包
        self.buffer.append((src, std, order))

    # 返回缓冲区中数据包数量
    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)


# 1.创建拓扑图，拓扑图中每个节点映射到一个对象。
# 2.每10ms进行一次传输，每个节点将自己的数据按照简单路径随机传递给下一个节点，直到非目标节点上没有数据为止。

def create_topo(draw=True):
    import networkx as nx
    import matplotlib.pyplot as plt

    # 创建一个空的无向图
    G = nx.Graph()

    # 添加节点
    for i in range(1,16):
        G.add_node(i)

    # 添加边
    G.add_edge(14, 3)
    G.add_edge(12, 1)
    G.add_edge(13, 2)

    G.add_edge(3,4)
    G.add_edge(1, 4)
    G.add_edge(1, 5)
    G.add_edge(1, 7)
    G.add_edge(2, 4)
    G.add_edge(2, 5)
    G.add_edge(2, 7)

    G.add_edge(4, 15)
    G.add_edge(4, 6)
    G.add_edge(5, 6)
    G.add_edge(5, 9)
    G.add_edge(7, 8)

    G.add_edge(6, 15)
    G.add_edge(6, 11)
    G.add_edge(8, 9)
    G.add_edge(9, 10)

    G.add_edge(10, 11)
    G.add_edge(11, 15)

    # 绘制图形
    if draw:
        pos = nx.spring_layout(G)  # 定义节点位置
        nx.draw(G,pos,with_labels=True,node_size=500,node_color="skyblue",
                font_size=20,font_weight="bold")
        plt.show()
    return G

import networkx as nx
import random

G = create_topo()

src_node_set = [12,13,14]
std_node_set = [10,11,15]

node_dicts = {}

capacity = 300+10
node_power = {}  # 用于存储节点的处理能力
high_node_set = [1,2,12,13]

for current_hop in range(1,16):
    if current_hop in src_node_set:
        node_dicts[current_hop] = Camera(capacity)
    elif current_hop in std_node_set:
        node_dicts[current_hop] = Data_center(capacity)
    else:
        node_dicts[current_hop] = Switch(capacity)
    if current_hop in high_node_set:
        node_power[current_hop] = 3
    else:
        node_power[current_hop] = 1

# 初始化
node_dicts[src_node_set[1]].add_dataset(src_node_set[1],std_node_set[1],50)


def random_find_next_hop(current_hop,std,G):
    """
    运用随机的方法寻找下一跳的位置
    :param current_hop: 当前的位置
    :param std: 目的地
    :param G: 图
    :return:
    """
    # 1.根据当前的位置和目的地获取图的所有简单路径
    all_paths = list(nx.all_simple_paths(G,source=current_hop,target=std))
    next_hop_list = [path[1] for path in all_paths]
    next_hop = random.choice(next_hop_list)
    return next_hop

temp_cache = []
count = 0  # 计数器
msg_num_dict = {}  # 信息量字典
# 统计初始的数据量
for current_hop in range(1,16):
    msg_num_dict[current_hop] = [node_dicts[current_hop].size()]

while True:
    # 各节点处理信息传出
    for current_hop in range(1,16):
        if current_hop not in std_node_set:  # 对所有非目的节点进行处理
            if node_dicts[current_hop].size() > 0:  # 判断节点中是否有数据，有数据才进行处理
                for _ in range(node_power[current_hop]):  # 根据处理能力进行传递
                    try:  # 防止出现空缓冲区pop的错误
                        msg = node_dicts[current_hop].trans()
                        src,std,index = msg
                        # 运用随机策略寻找下一跳的位置
                        next_hop = random_find_next_hop(current_hop,std,G)
                        # 创建一个临时缓存器，用于记录需要传递的信息
                        temp_cache.append((next_hop,msg))
                    except:
                        continue

    # 各节点处理信息传入
    if len(temp_cache)>0:
        for next_hop,msg in temp_cache:
            node_dicts[next_hop].add(*msg)
        else:
            temp_cache = []

    # 终止条件
    node_size_list = []  # 记录当前节点上信息量除了目的节点的
    for current_hop in range(1,16):
        if current_hop not in std_node_set:
            node_size_list.append(node_dicts[current_hop].size())

    # 记录所有节点上的信息量变化情况
    for current_hop in range(1, 16):
        msg_num_dict[current_hop].append(node_dicts[current_hop].size())

    if all([node_size==0 for node_size in node_size_list]):
        break
    else:
        count += 1

# 统计各节点数据量的变化
# 绘制各个节点上数据量变化曲线
import matplotlib.pyplot as plt
slot_time = list(range(count+2))
for current_hop in range(1,16):
    msg_num = msg_num_dict[current_hop]
    plt.figure(current_hop)

    # 绘制
    plt.plot(slot_time,msg_num)
    plt.xlabel("slot_time")
    plt.ylabel("msg_num")
    plt.title("msg numbers of node {}".format(current_hop))
    plt.show()















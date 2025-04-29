#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Auther:Evans_WP
@Contact:WenPeng19943251019@163.com
@Time:2024/3/27 10:07
@File:MARL_env.py
@Desc:这里的环境提供多智能体强化学习所要的数据，实现在线学习
"""
import collections
import xml.etree.ElementTree as ET
import networkx as nx
import matplotlib.pyplot as plt

import algorithms
from node_importance import NodeImportance
from data_trans import GetState
from comments.utils import random_simulation,re_popleft
import random
import logging
# from data_trans import add_node_attributes
import copy
# logging.basicConfig(level=logging.INFO)
import setting  # 自定义参数
import torch
from torch_geometric.utils.convert import from_networkx
import pickle
import numpy as np
from algorithms import DDPG
import statistics


MAX_TASK = setting.MAX_TASK  # 数据包总数k


# 对象区
# 1.摄像头
class Camera:
    # 产生数据，传递数据
    def __init__(self,id,capacity=MAX_TASK):
        """
        生成数据缓冲区
        :param id: 当前节点的编号
        :param capacity: 缓冲区的大小
        """
        self.id = id
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)

    # 生成待传输的数据，作为摄像头，自身的编号就是源节点
    def generated_data(self,std,num):
        """
        生成待传输的数据
        :param std: 目的地或目的地集
        :param num: 生成的数据数目，可以是向量
        :return:
        """
        src = self.id
        if isinstance(std,list):
            assert len(std) == len(num), "目的地个数必须和生成的数据种类一样多"
            # 后补随机生成数据的逻辑
        else:
            for order in range(1,num+1):
                precursor = None
                # 在信息中加入前驱，在每个交换机中添加本交换机的信息，信息的起始点为信息起始点的信息编号
                self.buffer.append((src,std,order,precursor))  # 注意这里先进的序号小的数据

    # 打乱buffer中的数据
    def buffer_resort(self):
        data_list = list(self.buffer)
        random.shuffle(data_list)
        self.buffer = collections.deque(data_list,maxlen=self.capacity)


    # 添加数据
    def add(self, src, std, order,precursor):
        # 每个交换机添加数据时，需要将自己的标签加入到前驱中
        self.buffer.append((src, std, order,precursor))

    # 传输数据
    def trans(self):
        """
        将数据传输出去
        :return:
        """
        # trans_data = self.buffer.popleft()
        trans_data = re_popleft(self.buffer)
        return trans_data

    # 观测状态
    def observe(self):
        """
        观测数据，不做传输
        :return:
        """

        # observe_data = self.buffer.popleft()
        observe_data = re_popleft(self.buffer)
        if observe_data[0] is not None:
            self.buffer.insert(0,observe_data)
        else:
            pass
        return observe_data


    # 重传机制
    def retrans(self,msg):
        """
        将数据包放回，进行重传
        :return:
        """
        self.buffer.insert(0,msg)


    # 返回缓存的数据包数量
    def buffersize(self):
        return len(self.buffer)

    # 返回缓存的数据包
    def size(self):
        return len(self.buffer)

# 2.交换机
class Switch:
    # 负责接收和传输数据
    def __init__(self,id,capacity=MAX_TASK):
        self.id = id
        self.buffer = collections.deque(maxlen=capacity)

    # 添加数据
    def add(self,src,std,order,precursor):
        self.buffer.append((src,std,order,precursor))

    # 传输数据
    def trans(self):
        """
        将数据传输出去
        :return:
        """
        # trans_data = self.buffer.popleft()  # 先进先出
        trans_data = re_popleft(self.buffer)
        return trans_data

    # 观测状态
    def observe(self):
        """
        观测数据，不做传输
        :return:
        """
        # observe_data = self.buffer.popleft()
        # self.buffer.insert(0, observe_data)
        observe_data = re_popleft(self.buffer)
        if observe_data[0] is not None:
            self.buffer.insert(0, observe_data)
        else:
            pass
        return observe_data

    # 重传机制
    def retrans(self, msg):
        """
        将数据包放回，进行重传
        :return:
        """
        self.buffer.insert(0, msg)

    # 返回缓存的数据包数量
    def buffersize(self):
        return len(self.buffer)

    # 返回缓存的数据包
    def size(self):
        return len(self.buffer)

# 3.数据中心
class DataCenter:
    # 收集数据，数据的存储包含两种形式，一种是缓存，类似于交换机，另一种是硬存(数据到达目标位置)
    def __init__(self, id, capacity=MAX_TASK):
        self.id = id
        self.buffer = collections.deque(maxlen=capacity)  # 缓存
        self.hard_buffer = collections.deque(maxlen=capacity)  # 硬存

    # 添加数据
    def add(self, src, std, order,precursor):  # 数据包包括源终点和第几个数据包
        if std == self.id:
            self.hard_buffer.append((src, std, order,precursor))
        else:
            self.buffer.append((src, std, order, precursor))

    # 传输数据
    def trans(self):
        """
        将数据传输出去
        :return:
        """
        # trans_data = self.buffer.popleft()  # 先进先出
        trans_data = re_popleft(self.buffer)
        return trans_data

    # 观测状态
    def observe(self):
        """
        观测数据，不做传输
        :return:
        """
        # observe_data = self.buffer.popleft()
        # self.buffer.insert(0, observe_data)
        observe_data = re_popleft(self.buffer)
        if observe_data[0] is not None:
            self.buffer.insert(0, observe_data)
        else:
            pass
        return observe_data

    # 重传机制
    def retrans(self, msg):
        """
        将数据包放回，进行重传
        :return:
        """
        self.buffer.insert(0, msg)

    # 返回缓存的数据包数量
    def buffersize(self):
        return len(self.buffer)

    # 返回缓冲区中总数据包数量
    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)+len(self.hard_buffer)

# 拓扑区
# 1.创建拓扑图，拓扑图中每个节点映射到一个对象。
# 2.每10ms进行一次传输，每个节点将自己的数据按照简单路径随机传递给下一个节点，直到非目标节点上没有数据为止。
# 3.将它写成可变拓扑，用.xml文件导入
class Topo:
    def __init__(self,xml_path='./topo.xml',traffic_matrix=setting.traffic_matrix_1):
        self.xml_path = xml_path
        self.traffic_matrix = traffic_matrix
        self.G,self.src_node_set,self.std_node_set,self.mid_node_set,self.node_power,\
            self.node_dicts,self.link_loss_rate = self._parse_xml()
        self.data_init()




    # 解析.xml成图
    def _parse_xml(self):
        tree = ET.parse(self.xml_path)  # 将.xml文件解析为tree
        root = tree.getroot()
        nodes_element = root.find("topology").find("nodes")
        links_element = root.find("topology").find("links")

        G = nx.Graph()  # 创建无向图
        src_node_set = []  # 源点集
        std_node_set = []  # 目标节点集
        mid_node_set = []  # 中间节点集
        node_power = {}  # 节点处理数据的能力，原文中c_i
        node_dicts = {}  # 节点映射，将topo节点和对象之间做映射
        link_loss_rate = {}  # 链路丢包率

        # 解析节点
        for child in nodes_element.iter():  # 遍历节点元素
            if child.tag == 'node':
                node_id = int(child.get('id'))
                G.add_node(node_id)
                if child.find('category').get('name') == "Camera":
                    src_node_set.append(node_id)
                    node_dicts[node_id] = Camera(node_id)
                elif child.find('category').get('name') == "Switch":
                    mid_node_set.append(node_id)
                    node_dicts[node_id] = Switch(node_id)
                elif child.find('category').get('name') == "DataCenter":
                    std_node_set.append(node_id)
                    node_dicts[node_id] = DataCenter(node_id)
                node_power[node_id] = int(child.find('power').get('ability'))


        # 解析链路
        for child in links_element.iter():  # 遍历边元素
            if child.tag == 'link':
                link_id = child.get('id')
                src,std = [int(node) for node in link_id.split('-')]
                G.add_edge(src,std)
                link_loss_rate[(src,std)] = float(child.find("loss").get('rate'))
                # 注意这个变量后面需要索引，所以正反指标都要，不然会报错，假设链路正反两个方向的丢包率一致
                link_loss_rate[(std,src)] = float(child.find("loss").get('rate'))

        # print(link_loss_rate)
        assert self.xml_path == setting.path_proof, f'{self.xml_path}设置有问题！setting.path_proof={setting.path_proof}'
        return G,src_node_set,std_node_set,mid_node_set,node_power,node_dicts,link_loss_rate


    def draw(self):
        # pos = nx.spring_layout(self.G)  # 定义节点位置
        # nx.draw(self.G, pos, with_labels=True, node_size=500, node_color="skyblue",
        #         font_size=20, font_weight="bold")
        # plt.show()
        # 将原点显示成菱形黄色      菱形：‘d’  黄色：‘y’
        # 将终点显示成六边形红色        六边形：‘H’  红色：‘r’
        # 其他节点显示成圆形蓝色        圆形：‘o’   蓝色：‘b’
        pos = nx.spring_layout(self.G)  # 定义节点位置
        src_node_set = self.src_node_set
        std_node_set = self.std_node_set
        node_shape = []
        node_color = []
        for node in self.G.nodes:
            if node in src_node_set:
                node_color.append('y')
            elif node in std_node_set:
                node_color.append('r')
            else:
                node_color.append('b')
        nx.draw(self.G,with_labels=True, node_size=500,
                node_color=node_color,font_size=20, font_weight="bold")
        plt.show()


    def data_init(self):
        # 按流量矩阵初始化
        assert len(self.src_node_set) == self.traffic_matrix.shape[0] and \
               len(self.std_node_set) == self.traffic_matrix.shape[1], \
            "流量矩阵行列数{}和源点数{}，目的点数{}对不上，请检查！".\
                format(self.traffic_matrix.shape,len(self.src_node_set),len(self.std_node_set))

        for row in range(self.traffic_matrix.shape[0]):
            for col in range(self.traffic_matrix.shape[0]):
                if self.traffic_matrix[row,col] != 0:
                    num = self.traffic_matrix[row,col]
                    src_node = self.src_node_set[row]
                    std_node = self.std_node_set[col]
                    self.node_dicts[src_node].generated_data(
                        std_node, num)

                    logging.debug("从节点{}发往节点{}数据包数为{}".format(src_node,
                                                             std_node, num))

        # 打乱数据顺序
        for row in range(self.traffic_matrix.shape[0]):
            src_node = self.src_node_set[row]
            self.node_dicts[src_node].buffer_resort()

class PacketRoutingEnv:
    def __init__(self, agents_node):
        self.topo = Topo(xml_path=setting.xml_path,traffic_matrix=setting.traffic_matrix)
        self.G = self.topo.G  # 原始图信息，仅仅包含节点和邻接矩阵
        self.node_package_num_dict = {}  # 用于保存每个节点上的待处理数据包量
        self.node_energy_use_dict = {}  # 用于保存每个节点上的能量使用量
        self.link_msg_dict = {}  # 用于保存每条链路上数据包的通过量
        self.temp_cache = []  # 用于保存临时缓存
        # 初始化topo，建立graph与实体之间的映射，初始化任务数据
        # 0. 通过节点重要性确定智能体的编号，数量
        self.agents_node = agents_node  # 多智能体的节点编号
        self.agents_num = len(self.agents_node)    # 多智能体的数量


        # 1. 求初始状态，需要注意的是这里是多智能体的状态，是多个状态的列表
        self.state = self.get_state()

        # 2. 添加每条链路上的通过数据包数的初始化
        for src,std in self.G.edges():
            self.link_msg_dict.setdefault((src,std),0)   # 双向皆有可能
            self.link_msg_dict.setdefault((std,src),0)

    def get_state(self):
        # 根据初始化topo，提取数据计算初始化状态，它是一个图的列表
        state0 = []
        # 其中的每一个图中包含topo信息，节点上的待处理数据包数和能量使用量
        # 0. 利用原始图获取初始状态
        original_state = GetState(self.G)
        # 1. 初始化待处理的数据包量和能量使用量
        for node in self.G.nodes:
            self.node_package_num_dict.setdefault(node,self.topo.node_dicts[node].size())
            self.node_energy_use_dict.setdefault(node,0)

        # 2.上面两个全局属性添加到图中获取全局状态
        node_attr_dict = {'package_num': self.node_package_num_dict, 'energy_use': self.node_energy_use_dict}
        original_state.up_global(node_attr_dict)

        # 3.对每个智能体初始化状态
        for node in self.agents_node:
            # try: # 防止出现空缓冲区pop的情况，这种情况发生局部状态为零
            #     # TODO：20240329
            #     msg = self.topo.node_dicts[node].observe()  # 智能体节点中获取数据包,20240331成观测,不做数据pop()
            #     src, std, index = msg
            #     current_state = copy.deepcopy(global_state)
            #     # TODO: 更改了一下data_trans.py中DataTrans类下的up_local方法
            #     current_state.up_local(node,std)
            #     state0.append(current_state)
            # except IndexError:
            #     current_state = copy.deepcopy(global_state)
            #     current_state.up_local(node)
            #     state0.append(current_state)
            msg = self.topo.node_dicts[node].observe()
            _, std, _, _ = msg  # 现在信息是四个元素
            current_state = copy.deepcopy(original_state)
            current_state.up_local(node,std)
            state0.append(current_state)
        return state0


    # TODO：在PacketRoutingEnv中新建step方法，用于获取智能体作用下的下一个状态和奖励函数
    def step(self,agents,explore=True):
        """
        通过这个函数获取下一个状态和奖励
        本来这个函数只要输入动作就可以了，通过动作获取相应的下一个状态，这里使用智能体作为变量，主要原因是根据能力不同，有的节点上可能需
        要进行多次传输，这个时候不仅仅只有一个动作，需要出现多个动作，只能将
        :param agents: 智能体字典
        :return:
        """
        done = False
        actions = []
        rewards = []
        agent_index = 0  # 用于索引智能体的位置
        # 备份数据用于计算奖励
        node_package_num_backup = copy.deepcopy(self.node_package_num_dict)  # 用于保存每个节点上的待处理数据包量
        node_energy_use_backup = copy.deepcopy(self.node_energy_use_dict)  # 用于保存每个节点上的能量使用量
        states = self.get_state()  # 获取状态
        # # 备份数据检验
        # print("备份数据检验")
        # print(node_package_num_backup)
        # print(node_energy_use_backup)

        for node in self.G.nodes:
            # 智能体节点，由智能体决策
            if node in self.agents_node:
                # TODO：将当前节点记录到状态中，方便求邻居节点给出动作，修改了一下data_trans.py中DataTrans类下的up_local方法
                # TODO: 写一个函数fun(state,agent)，返回该状态下的动作，不用写，这就是模型的中take_action,可以直接调用，但需要修改一下
                # TODO: 如果状态可以直接获取邻居节点，可以不用标识当前节点
                # TODO:20240331 加入观测后重写这段逻辑，可以参考仿真的写法
                # print("智能体索引检测",agent_index)
                agent = agents[agent_index]  # 提取智能体，智能体索引增加
                agent_reward = 0
                agent_action = torch.zeros(len(self.topo.G.nodes))
                for i, _ in enumerate(range(self.topo.node_power[node])):
                    # TODO: 20240401 加入传输机制
                    # 1. 获取信息的目标位置
                    # try:
                    #     msg = self.topo.node_dicts[node].trans()
                    #     _, std, _ = msg
                    #     # 2. 更新局部信息
                    #     self.state[agent_index].up_local(node,std)  # 保证状态，动作，智能体一一对应
                    #     # 3. 通过智能体获取动作
                    #     action = agent.take_action(self.state[agent_index])
                    #     if i == 0:
                    #         agent_action = action
                    #     assert sum(action) == 1,'动作不是独热编码'
                    #     # 4. 确定下一跳的位置
                    #     indice = np.where(action==1)
                    #     next_hop = self.agents_node[indice]
                    #     agent_reward = - node_package_num_backup[next_hop] - node_energy_energy_use_backup[next_hop]
                    #     # 5. 数据传输
                    #     self.tran_data(node,next_hop,msg)
                    # except IndexError:
                    #     continue
                    msg = self.topo.node_dicts[node].trans()
                    _, std, _, precursor = msg  # precursor 前驱节点
                    self.state[agent_index].up_local(node, std)
                    action = agent.take_action(self.state[agent_index],explore=explore)
                    indice = np.where(action.detach().cpu().view(-1).numpy() == 1)
                    next_hop = np.array(self.G.nodes)[indice][0]

                    # 随机进行一次采样
                    if i == 0:
                        agent_action = action
                        # agent_reward = - node_package_num_backup[next_hop] - setting.beta*node_energy_energy_use_backup[next_hop]
                        if not std:
                            agent_reward = 0
                        else:
                            agent_reward = reward_calculate(self.G,node,next_hop,std,node_package_num_backup,node_energy_use_backup)
                            if next_hop == precursor: # 下一个节点是前驱节点的话，需要给予惩罚
                                agent_reward += setting.loop_punishment  # 添加上回路惩罚
                    if std:  # 仅仅只有数据的情况下才传输数据，没有的情况下不用传输数据
                        # 传输数据前需要修改前驱
                        msg=(*msg[0:3],node)
                        self.tran_data(node, next_hop, precursor, msg)   # 能量的计算在数据传输中
                agent_index += 1  # 更新智能体位置
                actions.append(agent_action)  # 记录动作，用于采样
                rewards.append(agent_reward)


            else:  # 节点上没有智能体
                for _ in range(self.topo.node_power[node]):
                    # try:
                    #     msg = self.topo.node_dicts[node].trans()
                    #     _, std, _ = msg
                    #     next_hop = algorithms.random_short_path_hop(self.G,node,std)
                    #     self.tran_data(node,next_hop,msg)
                    # except IndexError:
                    #     continue
                    msg = self.topo.node_dicts[node].trans()
                    _, std, _, precursor = msg
                    next_hop = algorithms.avoid_loop_short_path(self.G, node, std, precursor)
                    if std:  # 仅仅只有数据的情况下才传输数据，没有的情况下不用传输数据
                        # 传输数据前需要修改前驱
                        msg=(*msg[0:3],node)
                        self.tran_data(node, next_hop, precursor, msg)
        # 6. 更新节点上的累积使用的能量
        if len(self.temp_cache) > 0:
            for next_hop, msg in self.temp_cache:
                self.topo.node_dicts[next_hop].add(*msg)
            else:
                self.temp_cache = []

        # 7. 更新节点上待处理的数据包信息
        for node in self.G.nodes:
            self.node_package_num_dict[node] = self.topo.node_dicts[node].buffersize()

        # 8. 终止条件
        if all([self.node_package_num_dict[node] == 0 for node in self.G.nodes]):
            if setting == 7:
                pass
            else:
                rewards = [rew+setting.terminal_reward for rew in rewards]
            done = True

        # 9. 下一个状态,修改类函数属性init_state为属性get_state,用于获取状态
        next_states = self.get_state()

        # 显示节点上数据变化情况
        # if not explore:
        #     print(node_package_num_backup,sum(node_package_num_backup.values()))
        # print(node_package_num_backup, sum(node_package_num_backup.values()))
        return states,actions,rewards,next_states,done

        # action = agents[node].take_action(self.state[index])
        # # 这里的动作是独热编码
        # assert sum(action) == 1,'动作不是独热编码'
        # # 单位节点的位置
        # indice = np.where(action == 1)
        # next_hop = self.agents_node[indice]
        # # TODO: 20240330
        # # 这可以作为当前节点的选择节点后的成本，进一步计算别的成本。
        # reward = - node_package_num_backup[next_hop] - node_energy_energy_use_backup[next_hop]
        # # 节点上的待处理的数据包数已经在之前得到了更新
        # # 下面更新节点上的使用过的能量数，这里涉及到传输成功或者失败，含有随机因子
        # node_redious_power = self.topo.node_power[node] - 1
        # # 数据包传输
        # if not random_simulation(self.topo.link_loss_rate[(node,next_hop)]):
        #     # 传输失败
        #     # 1. 增加发包消耗的能量
        #     self.node_energy_use_dict[node] += setting.transmission_energy
        #     # TODO: 20240331
        #     # 2. 将数据包放回,这里出现了问题，所有对象里还得添加观测功能，将观测和传输分开，这样能好处理一下。
        #     # TODO：在Camera，Switch和DataCenter三个对象中添加观测observe的功能
        #     self.topo.node_dicts[node].retrans(msg)
        # while node_redious_power>0:

    # TODO：在PacketRoutingEnv中新建step方法，用于获取智能体作用下的下一个状态和奖励函数
    def avoide_loop_step(self, agents, explore=True):
        """
        通过这个函数获取下一个状态和奖励,加入避回操作
        本来这个函数只要输入动作就可以了，通过动作获取相应的下一个状态，这里使用智能体作为变量，主要原因是根据能力不同，有的节点上可能需
        要进行多次传输，这个时候不仅仅只有一个动作，需要出现多个动作，只能将
        :param agents: 智能体字典
        :return:
        """
        done = False
        actions = []
        rewards = []
        precursors = []
        agent_index = 0  # 用于索引智能体的位置
        # 备份数据用于计算奖励
        node_package_num_backup = copy.deepcopy(self.node_package_num_dict)  # 用于保存每个节点上的待处理数据包量
        node_energy_use_backup = copy.deepcopy(self.node_energy_use_dict)  # 用于保存每个节点上的能量使用量
        states = self.get_state()  # 获取状态
        # # 备份数据检验
        # print("备份数据检验")
        # print(node_package_num_backup)
        # print(node_energy_use_backup)

        for node in self.G.nodes:
            # 智能体节点，由智能体决策
            if node in self.agents_node:
                # TODO：将当前节点记录到状态中，方便求邻居节点给出动作，修改了一下data_trans.py中DataTrans类下的up_local方法
                # TODO: 写一个函数fun(state,agent)，返回该状态下的动作，不用写，这就是模型的中take_action,可以直接调用，但需要修改一下
                # TODO: 如果状态可以直接获取邻居节点，可以不用标识当前节点
                # TODO:20240331 加入观测后重写这段逻辑，可以参考仿真的写法
                # print("智能体索引检测",agent_index)
                agent = agents[agent_index]  # 提取智能体，智能体索引增加
                agent_reward = 0
                agent_action = torch.zeros(len(self.topo.G.nodes))
                state_precursor = None
                for i, _ in enumerate(range(self.topo.node_power[node])):
                    # TODO: 20240401 加入传输机制
                    # 1. 获取信息的目标位置
                    # try:
                    #     msg = self.topo.node_dicts[node].trans()
                    #     _, std, _ = msg
                    #     # 2. 更新局部信息
                    #     self.state[agent_index].up_local(node,std)  # 保证状态，动作，智能体一一对应
                    #     # 3. 通过智能体获取动作
                    #     action = agent.take_action(self.state[agent_index])
                    #     if i == 0:
                    #         agent_action = action
                    #     assert sum(action) == 1,'动作不是独热编码'
                    #     # 4. 确定下一跳的位置
                    #     indice = np.where(action==1)
                    #     next_hop = self.agents_node[indice]
                    #     agent_reward = - node_package_num_backup[next_hop] - node_energy_energy_use_backup[next_hop]
                    #     # 5. 数据传输
                    #     self.tran_data(node,next_hop,msg)
                    # except IndexError:
                    #     continue
                    msg = self.topo.node_dicts[node].trans()
                    _, std, _, precursor = msg  # precursor 前驱节点
                    self.state[agent_index].up_local(node, std)
                    action = agent.avoid_loop_take_action(self.state[agent_index], precursor, explore=explore)
                    indice = np.where(action.detach().cpu().view(-1).numpy() == 1)
                    next_hop = np.array(self.G.nodes)[indice][0]
                    # if node == 2 and std:
                    #     print(f'节点{node}的前驱{precursor},下一个节点{next_hop},目标节点是{std}')
                    # if precursor == next_hop:ni
                    #     print(f'回路出现，当前节点为{node}，前驱节点为{precursor},下一个节点为{next_hop}')

                    # 随机进行一次采样
                    if i == 0:
                        agent_action = action
                        state_precursor = precursor
                        # agent_reward = - node_package_num_backup[next_hop] - setting.beta*node_energy_energy_use_backup[next_hop]
                        if not std:
                            agent_reward = 0
                        else:
                            agent_reward = reward_calculate(self.G, node, next_hop, std, node_package_num_backup,
                                                            node_energy_use_backup)
                            if next_hop == precursor:  # 下一个节点是前驱节点的话，需要给予惩罚
                                agent_reward += setting.loop_punishment  # 添加上回路惩罚
                    if std:  # 仅仅只有数据的情况下才传输数据，没有的情况下不用传输数据
                        # 传输数据前需要修改前驱
                        msg = (*msg[0:3], node)
                        self.tran_data(node, next_hop, precursor, msg)  # 能量的计算在数据传输中
                agent_index += 1  # 更新智能体位置
                precursors.append(state_precursor)
                actions.append(agent_action)  # 记录动作，用于采样
                rewards.append(agent_reward)

            else:  # 节点上没有智能体
                for _ in range(self.topo.node_power[node]):
                    # try:
                    #     msg = self.topo.node_dicts[node].trans()
                    #     _, std, _ = msg
                    #     next_hop = algorithms.random_short_path_hop(self.G,node,std)
                    #     self.tran_data(node,next_hop,msg)
                    # except IndexError:
                    #     continue
                    msg = self.topo.node_dicts[node].trans()
                    _, std, _, precursor = msg
                    next_hop = algorithms.avoid_loop_short_path(self.G, node, std, precursor)
                    # if precursor == next_hop:
                    #     print(f'回路出现，当前节点为{node}，前驱节点为{precursor},下一个节点为{next_hop}')
                    if std:  # 仅仅只有数据的情况下才传输数据，没有的情况下不用传输数据
                        # 传输数据前需要修改前驱
                        msg = (*msg[0:3], node)
                        self.tran_data(node, next_hop, precursor, msg)
        # 6. 更新节点上的累积使用的能量
        if len(self.temp_cache) > 0:
            for next_hop, msg in self.temp_cache:
                self.topo.node_dicts[next_hop].add(*msg)
            else:
                self.temp_cache = []

        # 7. 更新节点上待处理的数据包信息
        for node in self.G.nodes:
            self.node_package_num_dict[node] = self.topo.node_dicts[node].buffersize()

        # 8. 终止条件
        if all([self.node_package_num_dict[node] == 0 for node in self.G.nodes]):
            if setting == 7:
                pass
            else:
                rewards = [rew + setting.terminal_reward for rew in rewards]
            done = True

        # 9. 下一个状态,修改类函数属性init_state为属性get_state,用于获取状态
        next_states = self.get_state()

        # 显示节点上数据变化情况
        # if not explore:
        #     print(node_package_num_backup,sum(node_package_num_backup.values()))
        # print(node_package_num_backup, sum(node_package_num_backup.values()))
        return states, actions, rewards, next_states, done, precursors

        # action = agents[node].take_action(self.state[index])
        # # 这里的动作是独热编码
        # assert sum(action) == 1,'动作不是独热编码'
        # # 单位节点的位置
        # indice = np.where(action == 1)
        # next_hop = self.agents_node[indice]
        # # TODO: 20240330
        # # 这可以作为当前节点的选择节点后的成本，进一步计算别的成本。
        # reward = - node_package_num_backup[next_hop] - node_energy_energy_use_backup[next_hop]
        # # 节点上的待处理的数据包数已经在之前得到了更新
        # # 下面更新节点上的使用过的能量数，这里涉及到传输成功或者失败，含有随机因子
        # node_redious_power = self.topo.node_power[node] - 1
        # # 数据包传输
        # if not random_simulation(self.topo.link_loss_rate[(node,next_hop)]):
        #     # 传输失败
        #     # 1. 增加发包消耗的能量
        #     self.node_energy_use_dict[node] += setting.transmission_energy
        #     # TODO: 20240331
        #     # 2. 将数据包放回,这里出现了问题，所有对象里还得添加观测功能，将观测和传输分开，这样能好处理一下。
        #     # TODO：在Camera，Switch和DataCenter三个对象中添加观测observe的功能
        #     self.topo.node_dicts[node].retrans(msg)
        # while node_redious_power>0:

    # TODO:20240504
    # TODO：在PacketRoutingEnv中新建MARL_test_step,用于在环境中应用多智能体方法
    def MARL_test_step(self, agents, explore=True):
        """
        该函数用于测试的时候，按照MARL算法计算下一步，并统计完成时间。每条边上经过的数据量和能量使用量。
        :param agents:
        :return:
        """
        done = False
        agent_index = 0  # 用于索引智能体的位置
        for node in self.G.nodes:
            # 智能体节点，由智能体决策
            if node in self.agents_node:
                # TODO：将当前节点记录到状态中，方便求邻居节点给出动作，修改了一下data_trans.py中DataTrans类下的up_local方法
                # TODO: 写一个函数fun(state,agent)，返回该状态下的动作，不用写，这就是模型的中take_action,可以直接调用，但需要修改一下
                # TODO: 如果状态可以直接获取邻居节点，可以不用标识当前节点
                # TODO:20240331 加入观测后重写这段逻辑，可以参考仿真的写法
                agent = agents[agent_index]  # 提取智能体，智能体索引增加
                for i, _ in enumerate(range(self.topo.node_power[node])):
                    msg = self.topo.node_dicts[node].trans()
                    _, std, _, precursor = msg  # 但需要注意的是信息是四元组，而不是三元组
                    self.state[agent_index].up_local(node, std)
                    action = agent.avoid_loop_take_action(self.state[agent_index],precursor,explore=explore)
                    indice = np.where(action.detach().cpu().view(-1).numpy() == 1)
                    next_hop = np.array(self.G.nodes)[indice][0]  # 测试模块不用考虑去回路的问题，智能体在学习的情况下已经去掉了。
                    # if node == 6 and std:
                    #     print(f'节点{node}的前驱{precursor},下一个节点{next_hop},目标节点是{std}')
                    # if precursor == next_hop:
                    #     print(f'回路出现，当前节点为{node}，前驱节点为{precursor},下一个节点为{next_hop}')
                    if std:  # 仅仅只有数据的情况下才传输数据，没有的情况下不用传输数据
                        # 传输数据前需要修改前驱
                        msg = (*msg[0:3], node)
                        self.tran_data(node, next_hop, precursor, msg)
                agent_index += 1  # 更新智能体位置
            else:  # 节点上没有智能体
                for _ in range(self.topo.node_power[node]):
                    msg = self.topo.node_dicts[node].trans()
                    _, std, _ , precursor = msg
                    next_hop = algorithms.avoid_loop_short_path(self.G, node, std, precursor)
                    if precursor == next_hop:
                        print(f'回路出现，当前节点为{node}，前驱节点为{precursor},下一个节点为{next_hop}')
                    if std:  # 仅仅只有数据的情况下才传输数据，没有的情况下不用传输数据
                        # 传输数据前需要修改前驱
                        msg = (*msg[0:3], node)
                        self.tran_data(node, next_hop, precursor, msg)
        # 6. 更新数据
        if len(self.temp_cache) > 0:
            for next_hop, msg in self.temp_cache:
                self.topo.node_dicts[next_hop].add(*msg)
            else:
                self.temp_cache = []

        # 7. 更新节点上待处理的数据包信息
        for node in self.G.nodes:
            self.node_package_num_dict[node] = self.topo.node_dicts[node].buffersize()

        # 8. 终止条件
        if all([self.node_package_num_dict[node] == 0 for node in self.G.nodes]):
            done = True

        # 返回节点数据包数，节点能量累计使用量，链路累计数据包数，以及是否到达终点
        return self.node_package_num_dict,self.node_energy_use_dict,self.link_msg_dict,done

    # TODO：在PacketRoutingEnv中新建other_alg_step,用于在环境中应用传统算法
    def other_alg_step(self,algorithm):
        done = False
        for node in self.G.nodes:
            for _ in range(self.topo.node_power[node]):
                msg = self.topo.node_dicts[node].trans()
                _, std, _, precursor = msg
                next_hop = algorithm(self.G, node, std, precursor)
                if std:  # 仅仅只有数据的情况下才传输数据，没有的情况下不用传输数据
                    # 传输数据前需要修改前驱
                    msg = (*msg[0:3],node)
                    self.tran_data(node, next_hop, precursor, msg)
        # 6. 更新数据
        if len(self.temp_cache) > 0:
            for next_hop, msg in self.temp_cache:
                self.topo.node_dicts[next_hop].add(*msg)
            else:
                self.temp_cache = []

        # 7. 更新节点上待处理的数据包信息
        for node in self.G.nodes:
            self.node_package_num_dict[node] = self.topo.node_dicts[node].buffersize()

        # 8. 终止条件
        if all([self.node_package_num_dict[node] == 0 for node in self.G.nodes]):
            done = True

        # 返回节点数据包数，节点能量累计使用量，链路累计数据包数，以及是否到达终点
        return self.node_package_num_dict, self.node_energy_use_dict, self.link_msg_dict, done

    # TODO：在MARL_env.py中的类PacketRoutingEnv中添加node_importance_step方法来实现数据流向和节点重要性统计
    def node_importance_step(self,state,select_node):   # 这里的状态是结构是:[重要性节点集合，非重要性节点集合]
        importance_nodes_set = state[0]
        not_importance_nodes_set = state[1]
        rfph = algorithms.avoid_loop_random_path
        rsph = algorithms.avoid_loop_short_path
        done = False
        randomness = 0  # 如果不触发的话就为0
        for node in self.G.nodes:
            if node in importance_nodes_set and node == select_node:
                for _ in range(self.topo.node_power[node]):
                    msg = self.topo.node_dicts[node].trans()
                    _, std, _ , precursor = msg
                    next_hop,randomness0 = rsph(self.G, node, std, precursor, importance_statistic=True)  # 使用同样的算法
                    randomness += randomness0  # 可以决策数，相当于背包问题
                    if std:  # 仅仅只有数据的情况下才传输数据，没有的情况下不用传输数据
                        # 传输数据前需要修改前驱
                        msg=(*msg[0:3],node)
                        self.tran_data(node, next_hop, precursor, msg)
            elif node in not_importance_nodes_set and node == select_node:
                for _ in range(self.topo.node_power[node]):
                    msg = self.topo.node_dicts[node].trans()
                    _, std, _, precursor = msg
                    next_hop,randomness0 = rsph(self.G, node, std, precursor,importance_statistic=True)  # 使用同样的算法
                    randomness -= randomness0  # 可以决策数，相当于背包问题
                    if std:  # 仅仅只有数据的情况下才传输数据，没有的情况下不用传输数据
                        # 传输数据前需要修改前驱
                        msg=(*msg[0:3],node)
                        self.tran_data(node, next_hop, precursor, msg)
            else:
                for _ in range(self.topo.node_power[node]):
                    msg = self.topo.node_dicts[node].trans()
                    _, std, _ , precursor = msg
                    next_hop = rsph(self.G, node, std, precursor)
                    if std:  # 仅仅只有数据的情况下才传输数据，没有的情况下不用传输数据
                        # 传输数据前需要修改前驱
                        msg=(*msg[0:3],node)
                        self.tran_data(node, next_hop, precursor, msg)
        # 6. 更新数据
        if len(self.temp_cache) > 0:
            for next_hop, msg in self.temp_cache:
                self.topo.node_dicts[next_hop].add(*msg)
            else:
                self.temp_cache = []

        # 7. 更新节点上待处理的数据包信息
        for node in self.G.nodes:
            self.node_package_num_dict[node] = self.topo.node_dicts[node].buffersize()

        # 8. 终止条件
        if all([self.node_package_num_dict[node] == 0 for node in self.G.nodes]):
            done = True

        # 返回节点数据包数，节点能量累计使用量，链路累计数据包数，以及是否到达终点
        return randomness,done

    # TODO：在MARL_env.py中的类PacketRoutingEnv中添加node_importance_epsilon_step方法来实现数据流向和节点重要性统计,实现重要性节点上用epsilon贪心策略采样
    def node_importance_epsilon_step(self, state, select_node,epsilon):  # 这里的状态是结构是:[重要性节点集合，非重要性节点集合]
        importance_nodes_set = state[0]
        not_importance_nodes_set = state[1]
        rfph = algorithms.avoid_loop_random_path
        rsph = algorithms.avoid_loop_short_path
        done = False
        randomness = 0  # 如果不触发的话就为0
        for node in self.G.nodes:
            if node in importance_nodes_set and node == select_node:
                for _ in range(self.topo.node_power[node]):
                    msg = self.topo.node_dicts[node].trans()
                    # print(f'msg{msg}')
                    _, std, _, precursor = msg
                    next_hop0, randomness0 = rsph(self.G, node, std, precursor, importance_statistic=True)  # 使用同样的算法
                    next_hop1, randomness1 = rfph(self.G, node, std, precursor, importance_statistic=True)  # 用epsilon贪心策略选择其他节点
                    if random.random() < epsilon:
                        next_hop = next_hop1
                        # randomness += epsilon * randomness1
                    else:
                        next_hop = next_hop0
                        # randomness += (1 - epsilon) * randomness0
                    randomness += (1-epsilon)*randomness0 + epsilon*randomness1  # 可以决策数，相当于背包问题
                    if std:  # 仅仅只有数据的情况下才传输数据，没有的情况下不用传输数据
                        # 传输数据前需要修改前驱
                        msg=(*msg[0:3],node)
                        self.tran_data(node, next_hop, precursor, msg)
            elif node in not_importance_nodes_set and node == select_node:
                for _ in range(self.topo.node_power[node]):
                    msg = self.topo.node_dicts[node].trans()
                    _, std, _, precursor = msg
                    next_hop0, randomness0 = rsph(self.G, node, std, precursor, importance_statistic=True)  # 使用同样的算法
                    next_hop1, randomness1 = rfph(self.G, node, std, precursor, importance_statistic=True)  # 用epsilon贪心策略选择其他节点
                    # if random.random() < epsilon:
                    #     next_hop = next_hop1
                    #     # randomness -= epsilon * randomness
                    # else:
                    #     next_hop = next_hop0
                        # randomness -= (1 - epsilon) * randomness0
                    next_hop = next_hop0
                    randomness -= (1 - epsilon) * randomness0 + epsilon * randomness1  # 可以决策数，相当于背包问题
                    if std:  # 仅仅只有数据的情况下才传输数据，没有的情况下不用传输数据
                        # 传输数据前需要修改前驱
                        msg=(*msg[0:3],node)
                        self.tran_data(node, next_hop, precursor, msg)
            else:
                for _ in range(self.topo.node_power[node]):
                    msg = self.topo.node_dicts[node].trans()
                    _, std, _, precursor = msg
                    next_hop = rsph(self.G, node, std, precursor)
                    if std:  # 仅仅只有数据的情况下才传输数据，没有的情况下不用传输数据
                        # 传输数据前需要修改前驱
                        msg=(*msg[0:3],node)
                        self.tran_data(node, next_hop, precursor, msg)
        # 6. 更新数据
        if len(self.temp_cache) > 0:
            for next_hop, msg in self.temp_cache:
                self.topo.node_dicts[next_hop].add(*msg)
            else:
                self.temp_cache = []

        # 7. 更新节点上待处理的数据包信息
        for node in self.G.nodes:
            self.node_package_num_dict[node] = self.topo.node_dicts[node].buffersize()

        # 8. 终止条件
        if all([self.node_package_num_dict[node] == 0 for node in self.G.nodes]):
            done = True

        # 返回节点数据包数，节点能量累计使用量，链路累计数据包数，以及是否到达终点
        return randomness, done



    # TODO：在MARL_env.py中的类PacketRoutingEnv中添加state_importance_step方法来实现数据流向和状态重要性统计
    def state_importance_step(self,state):   # 这里的状态是结构是:[重要性节点集合，非重要性节点集合]
        importance_nodes_set = state[0]
        not_importance_nodes_set = state[1]
        rfph = algorithms.avoid_loop_random_path
        rsph = algorithms.avoid_loop_short_path
        done = False
        randomness = 0  # 如果不触发的话就为0
        for node in self.G.nodes:
            if node in importance_nodes_set:
                for _ in range(self.topo.node_power[node]):
                    msg = self.topo.node_dicts[node].trans()
                    _, std, _, precursor = msg
                    next_hop,randomness0 = rsph(self.G, node, std, precursor, importance_statistic=True)  # 使用同样的算法
                    randomness += randomness0  # 可以决策数，相当于背包问题
                    if std:  # 仅仅只有数据的情况下才传输数据，没有的情况下不用传输数据
                        # 传输数据前需要修改前驱
                        msg=(*msg[0:3],node)
                        self.tran_data(node, next_hop, precursor, msg)
            else:
                for _ in range(self.topo.node_power[node]):
                    msg = self.topo.node_dicts[node].trans()
                    _, std, _, precursor = msg
                    next_hop = rsph(self.G, node, std, precursor)
                    if std:  # 仅仅只有数据的情况下才传输数据，没有的情况下不用传输数据
                        # 传输数据前需要修改前驱
                        msg=(*msg[0:3],node)
                        self.tran_data(node, next_hop, precursor, msg)
        # 6. 更新数据
        if len(self.temp_cache) > 0:
            for next_hop, msg in self.temp_cache:
                self.topo.node_dicts[next_hop].add(*msg)
            else:
                self.temp_cache = []

        # 7. 更新节点上待处理的数据包信息
        for node in self.G.nodes:
            self.node_package_num_dict[node] = self.topo.node_dicts[node].buffersize()

        # 8. 终止条件
        if all([self.node_package_num_dict[node] == 0 for node in self.G.nodes]):
            done = True

        # 返回节点数据包数，节点能量累计使用量，链路累计数据包数，以及是否到达终点
        return randomness,done


    # TODO: 在PacketRoutingEnv下建立一个tran_data方法，用于传输数据
    def tran_data(self,node,next_hop,precursor,msg):
        """
        传输数据的方法
        :param node: 当前节点
        :param next_hop: 下一个节点
        传输过程中更新能量使用量，传输完更新节点待处理的数据包数量,更新数据缓存
        """
        # if node == 4 and next_hop==15:
        #     print(next_hop)
        _,std,_,_ = msg
        if msg is not None:
            if not random_simulation(self.topo.link_loss_rate[(node,next_hop)]):
                # 传输失败
                # 新增传输数据包传输能量统计
                self.node_energy_use_dict[node] += setting.transmission_energy
                # 进入重传
                msg = (*msg[0:3], precursor)
                self.topo.node_dicts[node].retrans(msg)
            else:
                # 传输成功
                # 新增传输数据包的能量统计
                self.node_energy_use_dict[node] += setting.transmission_energy
                self.node_energy_use_dict[next_hop] += setting.receiving_energy
                self.link_msg_dict[(node,next_hop)] += 1
                self.temp_cache.append((next_hop,msg))
        else:
            pass

    # TODO: 在PacketRoutingEnv下建立一个reset方法，用于将状态
    def reset(self):
        """
            回归初始状态
        """
        self.topo = Topo(xml_path=setting.xml_path,traffic_matrix=setting.traffic_matrix)
        self.node_package_num_dict = {}  # 用于保存每个节点上的待处理数据包量
        self.node_energy_use_dict = {}  # 用于保存每个节点上的能量使用量
        # 1. 求初始状态，需要注意的是这里是多智能体的状态，是多个状态的列表
        self.state = self.get_state()

def reward_calculate(G,cur_node,next_hop,std_node,node_package_num,node_energy_num):
    """
    计算奖励函数
    :param G: 图
    :param cur_node: 当前节点
    :param next_hop: 下一节点
    :param std_node: 目标节点
    :param node_package_num: 节点上的数据包数量
    :param node_energy_num: 节点上的能量使用量
    :return: 奖励
    """
    # 1. 计算当前节点到目的节点的最短跳数
    cur_cost = len(nx.shortest_path(G, source=cur_node, target=std_node)) - 1
    # 2. 计算下一个节点到目的节点的最短跳数
    next_cost = len(nx.shortest_path(G, source=next_hop, target=std_node)) - 1
    # 3. 计算邻居节点中累计处理最大和最小的数据包和累计能量使用数
    neighbor_nodes = nx.neighbors(G, cur_node)
    for node in neighbor_nodes:
        if 'min_package' not in locals():
            min_package = node_package_num[node]
        else:
            min_package = min(node_package_num[node], min_package)
        if 'max_package' not in locals():
            max_package = node_package_num[node]
        else:
            max_package = max(node_package_num[node], max_package)
        if 'min_energy' not in locals():
            min_energy = node_energy_num[node]
        else:
            min_energy = min(node_energy_num[node], min_energy)
        if 'max_energy' not in locals():
            max_energy = node_energy_num[node]
        else:
            max_energy = max(node_energy_num[node], max_energy)

    next_package = node_package_num[next_hop]
    next_energy = node_energy_num[next_hop]

    cost = 0
    if setting.reward_type == 1:
        # 4. 返回奖励函数
        cost = (cur_cost - next_cost) - 0.5*(next_package-min_package)/(max_package-min_package+setting.epsilon) - 0.3*(next_energy-
                min_energy)/(max_energy-min_energy+setting.epsilon)
    elif setting.reward_type == 2:
        cost = - 0.5*(next_package-min_package)/(max_package-min_package+setting.epsilon) - 0.5*(next_energy-
                min_energy)/(max_energy-min_energy+setting.epsilon)
    elif setting.reward_type == 3:
        cost = -sum(node_package_num.values())
    elif setting.reward_type == 4:
        cost = -sum(node_package_num.values())-0.001*max(node_energy_num.values())
    elif setting.reward_type == 5:
        cost = -sum(node_package_num.values())-0.001*(max(node_energy_num.values())-min(node_energy_num.values()))
    elif setting.reward_type == 6:
        cost = -sum(node_package_num.values())/MAX_TASK - (max(node_energy_num.values())
                        - min(node_energy_num.values()))/((setting.transmission_energy+setting.receiving_energy)*MAX_TASK)
    elif setting.reward_type == 7:
        cost = (setting.pre_state-sum(node_package_num.values()))/MAX_TASK*setting.terminal_reward - (max(node_energy_num.values())
                        - min(node_energy_num.values()))/((setting.transmission_energy+setting.receiving_energy)*MAX_TASK)
        setting.pre_state = sum(node_package_num.values())
    elif setting.reward_type == 8:
        cost = - setting.beta * sum(node_package_num.values())/MAX_TASK - (1-setting.beta)*(max(node_energy_num.values())
                        - min(node_energy_num.values()))/max(node_energy_num.values())
    elif setting.reward_type == 9:
        cost = -1
    elif setting.reward_type == 10:
        node_energy_max = max(node_energy_num.values())
        node_energy_min = min(node_energy_num.values())
        node_energy_mean = statistics.mean(node_energy_num.values())
        cost = - setting.beta * sum(node_package_num.values())/MAX_TASK - (1-setting.beta)*(abs(node_energy_num[cur_node]-node_energy_mean))/max([
            node_energy_max-node_energy_mean,node_energy_mean-node_energy_min])
        assert cost < 0, f'成本函数值{cost}不小于零，检查bug!'
    elif setting.reward_type == 11:
        node_energy_max = max(node_energy_num.values())
        node_energy_min = min(node_energy_num.values())
        node_energy_mean = statistics.mean(node_energy_num.values())
        cost = - setting.beta * sum(node_package_num.values())/MAX_TASK - (1-setting.beta)*(abs(node_energy_num[cur_node]+node_energy_num[next_hop]-2*node_energy_mean))/max([
            node_energy_max-node_energy_mean,node_energy_mean-node_energy_min])
        assert cost < 0, f'成本函数值{cost}不小于零，检查bug!'
    elif setting.reward_type == 12:
        # node_energy_max = max(node_energy_num.values())
        # node_energy_min = min(node_energy_num.values())
        # node_energy_mean = statistics.mean(node_energy_num.values())
        node_energy_list = np.array(list(node_energy_num.values()))
        cost = - setting.beta * sum(node_package_num.values()) / MAX_TASK - (1 - setting.beta) * (np.var(node_energy_list)/(
                                                                                                  np.ptp(node_energy_list)**2+1e-8))
        cost = float(cost)
        assert cost < 0, f'成本函数值{cost}不小于零，检查bug!'
    return cost



if __name__ == '__main__':
    # 智能体节点编号
    agents_node = [1,2,4,5,6]
    agents_num = len(agents_node)
    state_dim = 4
    critic_input_dim = (state_dim+1)*agents_num
    critic_hidden_dim = agents_num
    actor_lr = 0.0001
    critic_lr = 0.000001
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = PacketRoutingEnv(agents_node)

    agents = [DDPG(node_id,state_dim,critic_input_dim,critic_hidden_dim,actor_lr,critic_lr,device) for node_id in agents_node]

    # actions, rewards, next_states, done = env.step(agents)
    # TODO：新加数据包结构，重写封装.popleft()功能，简化获取数据包的方法
    # TODO：20240402 星期二
    # TODO：修改data_trans.py里Get_state类中normalization方法，避免出现分母为零的状态。
    # TODO: 邻接矩阵需要转化为稀疏矩阵，并进行重排。这样比较麻烦，直接写成稠密矩阵的模式。
    # TODO: 修改了algorithms.py下DDPG中take_action的输出，这里可能为后面带来问题。
    # TODO: 修改了algorithms.py下random_short_path_hop函数，使得它可以适应没有数据的形式。
    # TODO: 20240403 星期三
    # TODO：修改agorithms.py下DDPG类中onehot_from_logits方法，用于处理最大动作不仅仅只有一个的情况，倾向于选择后面一个。
    # TODO: 在MARL_env.py中Camera类中添加add方法，用于处理数据发送失败的情况。
    # print('动作',actions)
    # print('收益',rewards)
    # print('下一个动作',next_states)
    # print('结束状态',done)
    # actions, rewards, next_states, done = env.step(agents)
    # print('动作', actions)
    # print('收益', rewards)
    # print('下一个动作', next_states)
    # print('结束状态', done)
    #
    # actions, rewards, next_states, done = env.step(agents)
    # print('动作', actions)
    # print('收益', rewards)
    # print('下一个动作', next_states)
    # print('结束状态', done)

    # TODO：经过测试，随着能量的累积，成本越来越高，这似乎是不合理的。
    # TODO：该测试只是在非探索的情况下使用的，如果是探索，还需进一步进行测试。
    done = False
    explore = True
    count = 0
    sample_num = 20
    while not done:
        states,actions, rewards, next_states, done = env.step(agents,explore)
        print('状态',states)
        print('动作',actions)
        print('收益',rewards)
        print('下一个动作',next_states)
        print('结束状态',done)
        sample_data = [states,actions, rewards, next_states, done]

        count += 1
        if count == sample_num:
            with open('sample.pkl', 'wb') as f:
                pickle.dump(sample_data, f)
            break

    # TODO: 20240405 星期五
    # TODO: 在data_trans.py中GetState类中添加__getstate__和__setstate__方法，确保可以存储。
    # TODO：修改MARL_env.py中PacketRoutingEnv类中的step方法下的输出，让输出中包含状态。
    # TODO: 在setting中删除了algorithm的调用，排除了algorithm和setting循环调用的bug。




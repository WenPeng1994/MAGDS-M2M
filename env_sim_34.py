#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Auther:Evans_WP
@Contact:WenPeng19943251019@163.com
@Time:2024/3/1 15:30
@File:env_sim_31.py
@Desc:带有缓冲区通信仿真
"""
"""
通信环境包括三种对象：
第一种是产生数据的摄像头(Camera),其功能是产生数据，并将数据传输出去
第二种是传递数据的交换机(Switch),其功能是转发数据，负责接收并将数据发给下一个设备。
第三种是收集数据的数据中心(Data_center),其功能是存储数据，同时具有交换机的功能可以转存数据
"""
"""
本实验仿真的环境可以适应多摄像头，多数据中心的情形
"""
import collections
import xml.etree.ElementTree as ET
import networkx as nx
import matplotlib.pyplot as plt
import random
import logging
logging.basicConfig(level=logging.INFO)
import setting  # 自定义参数
from torch_geometric.utils.convert import from_networkx

MAX_TASK = 300  # 原文中的k

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
                self.buffer.append((src,std,order))  # 注意这里先进的序号小的数据

    # 传输数据
    def trans(self):
        """
        将数据传输出去
        :return:
        """
        trans_data = self.buffer.popleft()
        return trans_data


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
    def add(self,src,std,order):
        self.buffer.append((src,std,order))

    # 传输数据
    def trans(self):
        """
        将数据传输出去
        :return:
        """
        trans_data = self.buffer.popleft()  # 先进先出
        return trans_data

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
    def add(self, src, std, order):  # 数据包包括源终点和第几个数据包
        if std == self.id:
            self.hard_buffer.append((src, std, order))
        else:
            self.buffer.append((src, std, order))

    # 传输数据
    def trans(self):
        """
        将数据传输出去
        :return:
        """
        trans_data = self.buffer.popleft()  # 先进先出
        return trans_data

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
                    logging.info("从节点{}发往节点{}数据包数为{}".format(src_node,
                                                             std_node, num))

        # num = 300
        # src_node = self.src_node_set[1]
        # std_node = self.std_node_set[1]
        # self.node_dicts[src_node].generated_data(
        #     std_node, num)
        # logging.info("从节点{}发往节点{}数据包数为{}".format(src_node,
        #                                          std_node,num))


    # def random_find_next_hop(self,current_hop,std):
    #     """
    #     运用随机的方法寻找下一跳的位置
    #     :param current_hop: 当前的位置
    #     :param std: 目的地
    #     :return:
    #     """
    #     # 1.根据当前的位置和目的地获取图的所有简单路径
    #     all_paths = list(nx.all_simple_paths(self.G,source=current_hop,target=std))
    #     next_hop_list = [path[1] for path in all_paths]
    #     next_hop = random.choice(next_hop_list)
    #     return next_hop

def run(topo):
    temp_cache = []
    count = 0  # 计数器，经过的跳数，与完成传输的时间成正比
    msg_num_dict = {}  # 信息量字典
    link_msg_dict = {}  # 链路上信息量字典，包含单个目的地的信息量，和总的通过信息的量
    node_energy_use_dict = {}  # 节点上能量使用字典，计算每个节点上的能量使用情况
    # {(current_hop,next_hop):{(src,std):num,...,'total':num}}
    # 统计初始的数据量
    for current_hop in topo.G.nodes:
        msg_num_dict[current_hop] = [topo.node_dicts[current_hop].size()]
    while True:
        # 各节点处理信息传出
        for current_hop in topo.G.nodes:
            if topo.node_dicts[current_hop].buffersize() > 0:  # 判断节点中是否有数据，有数据才进行处理
                for _ in range(topo.node_power[current_hop]):  # 根据处理能力进行传递
                    try:  # 防止出现空缓冲区pop的错误
                        msg = topo.node_dicts[current_hop].trans()
                        src, std, index = msg
                        # 运用随机策略寻找下一跳的位置
                        # next_hop = topo.random_find_next_hop(current_hop, std)
                        # next_hop = algorithms.random_find_next_hop(topo.G,current_hop,std)
                        next_hop = setting.use_algorithms(topo.G,current_hop,std)
                        # 加入重传机制,加入时间限制
                        if not random_simulation(topo.link_loss_rate[(current_hop,next_hop)]):
                            # 传输失败
                            # 1. 记录传输失败的次数
                            link_msg_dict.setdefault((current_hop, next_hop), dict())  # 初始化字典
                            link_msg_dict[(current_hop,next_hop)].setdefault("loss_packet",0)
                            link_msg_dict[(current_hop,next_hop)]["loss_packet"] += 1
                            # 2. 增加发包消耗的能量统计
                            node_energy_use_dict.setdefault(current_hop, 0)
                            node_energy_use_dict[current_hop] += setting.transmission_energy  # 传输失败，传输能量是要消耗的
                            # 3. 将包放回进行重传
                            topo.node_dicts[current_hop].retrans(msg)
                        else:
                            # 传输成功
                            # 创建一个临时缓存器，用于记录需要传递的信息
                            temp_cache.append((next_hop, msg))
                            link_msg_dict.setdefault((current_hop, next_hop), dict())  # 初始化字典
                            link_msg_dict[(current_hop,next_hop)].setdefault((src,std),0)
                            link_msg_dict[(current_hop, next_hop)][(src,std)] += 1  # 如果传输失败不计数
                            link_msg_dict[(current_hop,next_hop)].setdefault('total',0)
                            link_msg_dict[(current_hop,next_hop)]['total'] += 1  # 如果传输失败不计数
                            node_energy_use_dict.setdefault(current_hop,0)
                            node_energy_use_dict[current_hop] += setting.transmission_energy  # 传输失败，传输能量是要消耗的
                            node_energy_use_dict.setdefault(next_hop,0)
                            node_energy_use_dict[next_hop] += setting.receiving_energy  # 传输失败，接收能力不计算
                        # 创建一个临时缓存器，用于记录需要传递的信息
                        # temp_cache.append((next_hop, msg))
                        # link_msg_dict.setdefault((current_hop, next_hop), dict())  # 初始化字典
                        # link_msg_dict[(current_hop,next_hop)].setdefault((src,std),0)
                        # link_msg_dict[(current_hop, next_hop)][(src,std)] += 1  # 如果传输失败不计数
                        # link_msg_dict[(current_hop,next_hop)].setdefault('total',0)
                        # link_msg_dict[(current_hop,next_hop)]['total'] += 1  # 如果传输失败不计数
                        # node_energy_use_dict.setdefault(current_hop,0)
                        # node_energy_use_dict[current_hop] += setting.transmission_energy  # 传输失败，传输能量是要消耗的
                        # node_energy_use_dict.setdefault(next_hop,0)
                        # node_energy_use_dict[next_hop] += setting.receiving_energy  # 传输失败，接收能力不计算
                    # except Exception as e:
                    #     print("发生了一个未知的错误：",type(e).__name__,":",e)
                    #     continue
                    except IndexError:
                        continue
        # 各节点处理信息传入
        if len(temp_cache) > 0:
            for next_hop, msg in temp_cache:
                topo.node_dicts[next_hop].add(*msg)
            else:
                temp_cache = []

        # 终止条件
        node_size_list = []  # 记录当前节点上信息量除了目的节点的
        for current_hop in topo.G.nodes:
            node_size_list.append(topo.node_dicts[current_hop].buffersize())

        # 记录所有节点上的信息量变化情况
        for current_hop in topo.G.nodes:
            msg_num_dict[current_hop].append(topo.node_dicts[current_hop].size())

        if all([node_size == 0 for node_size in node_size_list]):  # 终止条件
            break
        else:
            count += 1
    return msg_num_dict,count,link_msg_dict,node_energy_use_dict

# 随机概率模拟
def random_simulation(rate):
    """
    随机概率模拟
    :param rate: 丢包率
    :return:
    """
    if random.random() < 1-rate:  # 1-rate 传输成功
        return True  # 传输成功
    else:
        return False  # 传输失败

# 性能展示
def performance_display(topo,msg_num_dict,count,link_msg_dict,node_energy_use_dict):
    # 统计各节点数据量的变化
    # 绘制各个节点上数据量变化曲线
    slot_time = list(range(count + 2))
    for current_hop in topo.G.nodes:
        msg_num = msg_num_dict[current_hop]
        plt.figure(current_hop)

        # 绘制
        plt.plot(slot_time, msg_num)
        plt.xlabel("slot_time")
        plt.ylabel("msg_num")
        plt.title("msg numbers of node {}".format(current_hop))
        plt.show()

    # 显示链路上经过的数据量信息
    link_list = []
    link_msg = []
    for item in link_msg_dict.keys():
        link_list.append(str(item))
        link_msg.append(link_msg_dict[item]['total'])
    plt.figure(figsize=(8,6))
    plt.barh(link_list,link_msg)
    for i,num in enumerate(link_msg):
        plt.text(num,i,str(num),ha='left',va='center')
    plt.xlabel('msg number of link')
    plt.ylabel('link')
    plt.title('Information statistics on links')
    plt.tight_layout()  # 自动调整子图参数
    plt.show()

    # 显示节点上能量使用情况
    node_list = []
    node_energy_use_list = []
    for node,energy in node_energy_use_dict.items():
        node_list.append(str(node))
        node_energy_use_list.append(energy)
    plt.figure(figsize=(8,6))
    plt.barh(node_list,node_energy_use_list)
    for i,num in enumerate(node_energy_use_list):
        plt.text(num,i,str(num),ha='left',va='center')
    plt.xlabel('used energy of node')
    plt.ylabel('node')
    plt.title("Node energy usage statistics")
    plt.tight_layout()  # 自动调整子图参数
    plt.show()


if __name__ == '__main__':
    topo = Topo(traffic_matrix=setting.traffic_matrix_4)
    # print(topo.G.nodes)
    msg_num_dict, count, link_msg_dict,node_energy_use_dict = run(topo)
    # # print(msg_num_dict,count)
    # # print(link_msg_dict)
    # performance_display(topo,msg_num_dict,count,link_msg_dict,node_energy_use_dict)
    topo.draw()
    # from node_importance import NodeImportance
    # node_importance_measure = NodeImportance(topo.G)
    # node_importance_measure.node_centrality_analyze()
    pyg_data = from_networkx(topo.G)
    print(pyg_data)
    print('调试')








    """ 20240302任务
    1. 实现流量矩阵(traffic matrix)的仿真  20240302
    2. 实现链路上经过数据量的统计   20240303
    3. 实现剩余能量统计 
    能量使用统计，每个节点发送一个数据包能量为105mW，接收一个数据包能量为54mW，模型待机30mW 20240304
    4. 实现流量负载率的计算
    链路的流量负载除以总流量
    5. 实现链路丢包率的计算，丢包后重新发送，发送的能量有损耗，接收的能量无损耗 20240305
    """
    """20240304任务
    对比实验
    MAPPO
    SPR：最短路径路由
    Q-routing[17]
    QELAR[20]
    DQRC[33]
    """
    """
    计算各种中心性看相应的结果 20240310
    设置一个比例将前几个重要的指标取出来作为重要节点 20240311
    """
    """20240319
    解决算法名称代入的问题
    """
    """
    选择强化学习算法
    GCN 20240321
    1. 暂时仅考虑节点上的属性
    
    """














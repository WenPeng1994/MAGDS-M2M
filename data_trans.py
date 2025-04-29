#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Auther:Evans_WP
@Contact:WenPeng19943251019@163.com
@Time:2024/3/20 21:55
@File:data_trans.py
@Desc:实现将Networkx信息转化成Geometric数据
"""
import copy
import networkx as nx
from collections import namedtuple
import numpy as np
import torch
import pickle

# 定义各种数据操作
import setting


def add_node_attributes(G,node_attr_dict):
    """
    添加节点属性
    :param G: 图
    :param node_attr_dict: 节点属性结合，{attr_name,node_dict}
    :return: new_G: 添加了属性的新的图
    """
    new_G = copy.deepcopy(G)
    node_attributes = {}
    for node in new_G.nodes:
        for attr in node_attr_dict.keys():
            node_attributes.setdefault(node, dict())
            node_attributes[node][attr] = node_attr_dict[attr][node]

    nx.set_node_attributes(new_G,node_attributes)
    return new_G

#     # new_G = copy.deepcopy(G)
#     # node_attributes = {}
#     # attr_name = []
#     # for node in new_G.nodes:
#     #     for attr in node_attr_dict.keys():
#     #         node_attributes.setdefault(node, dict())
#     #         node_attributes[node].setdefault('x',list())
#     #         node_attributes[node]['x'].append(node_attr_dict[attr][node])
#     # nx.set_node_attributes(new_G,node_attributes)
#     # return new_G,attr_name
#
#
# class DataTrans:
#     """
#     networkx 中的数据和pyg中的数据装换
#     """
#     def __init__(self,G):
#         self.G = G
#         self.Data = namedtuple('Data',['x','adjacency','node_id'])
#         self.num_nodes = self.G.number_of_nodes()
#         self.node_id = np.array(self.G.nodes)
#
#
#
#     def to_Data(self):
#         """
#         将数据转化为Data
#         :return:
#         """
#         # 1.获取节点ID
#         node_id = self.node_id
#         # 2.获取节点属性x
#         node_attributes = dict(self.G.nodes(data=True))  # 获取图的属性字典
#         attr_name_list = list(node_attributes[node_id[0]].keys())  # 这里的0是必要的，表示索引0号节点，可能没有节点，遇到了后面处理
#         for attr_name in attr_name_list:
#             attrs = nx.get_node_attributes(self.G,attr_name).values()  # 取相应属性的所有值
#             if 'x' not in locals():
#                 x = [list(attrs)]
#             else:
#                 x = np.concatenate((x,[list(attrs)]),axis=0)
#         # print(x.T)
#         # 3.获取邻接矩阵
#         src_list = []
#         std_list = []
#         for src,std in self.G.edges():
#             src_list.extend([src,std])
#             std_list.extend([std,src])
#         adjacency = np.concatenate(([src_list],[std_list]),axis=0)
#         # print(adjeacency)
#         return self.Data(x=x, adjacency=adjacency, node_id=node_id)
#
#     def get_neighbors(self,node):
#         """
#         获取邻居节点和邻居节点热索引
#         :param node:
#         :return:
#         """
#         neighbors = list(self.G.neighbors(node))
#         neighbor_hot = np.zeros(self.num_nodes,dtype=bool)
#         for index, node in enumerate(self.node_id):
#             if node in neighbors:
#                 neighbor_hot[index] = True
#         # print(neighbor_hot)
#         return neighbors,neighbor_hot


class GetState:
    def __init__(self,G):
        self.G = G
        self.node_id = 0  # 当前节点的id，如果是0，表示还没有更新
        self.Data = setting.Data
        self.node_ids = np.array(self.G.nodes)
        self.node_num = len(self.G.nodes)
        self.epsilon = 1e-3


    # 对全局属性进行归一化
    def normalization(self,node_attr_dict):
        # 对节点属性进行归一化
        norm_node_attr_dict = copy.deepcopy(node_attr_dict)
        for key,value in node_attr_dict.items():
            norm_value = list(np.array(list(value.values()))/
                              (np.sum(np.array(list(value.values())))+self.epsilon))
            for i,node in enumerate(value.keys()):
                norm_node_attr_dict[key][node] = norm_value[i]
        return norm_node_attr_dict


    # 全局信息更新，更新全局变量时，一般需要进行归一化
    def up_global(self,node_attr_dict,normalize=True):
        new_G = copy.deepcopy(self.G)
        node_attributes = {}
        if normalize:
            node_attr_dict = self.normalization(node_attr_dict)  # 对全局属性进行归一化
        for node in new_G.nodes:
            for attr in node_attr_dict.keys():
                node_attributes.setdefault(node, dict())
                node_attributes[node][attr] = node_attr_dict[attr][node]

        nx.set_node_attributes(new_G, node_attributes)
        self.G = new_G  # 更新图


    # 局部信息更新，获得每个智能体观测的状态
    # 局部信息指的是，当前数据包的信息，之前版本考虑了原点和目标点，这里修改为仅仅考虑目标点。
    # 这个状态变成独热编码,添加了标识不能进行pop数据包的情况
    def up_local(self,src,std=None):   # 没有值，用于标识不能进行pop数据包的情况
        self.node_id = src
        new_G = copy.deepcopy(self.G)
        node_attributes = {}
        for node in new_G.nodes:
            node_attributes.setdefault(node,dict())
            if node == src:
                node_attributes[node]['src'] = 1
            else:
                node_attributes[node]['src'] = 0
            if std:
                if node == std:
                    node_attributes[node]['std'] = 1
                else:
                    node_attributes[node]['std'] = 0
            else:
                node_attributes[node]['std'] = 0


        nx.set_node_attributes(new_G,node_attributes)
        self.G = new_G  # 更新图


    # def up_local(self,src,std,tag=False):  # tag用于标记缓冲区
    #     self.node_id = src
    #     new_G = copy.deepcopy(self.G)
    #     node_attributes = {}
    #     for node in new_G.nodes:
    #         node_attributes.setdefault(node,dict())
    #         if node == src:
    #             node_attributes[node]['src_std'] = -1
    #         elif node == std:
    #             node_attributes[node]['src_std'] = 1
    #         else:
    #             node_attributes[node]['src_std'] = 0
    #
    #     nx.set_node_attributes(new_G,node_attributes)
    #     self.G = new_G  # 更新图


    # 行为动作更新
    def up_act(self,one_hot_act):  # one_hot_act 是下一跳的独热编码
        new_G = copy.deepcopy(self.G)
        node_attributes = {}
        assert self.node_id, "请先进行观测"
        for i,node in enumerate(new_G.nodes):
            node_attributes.setdefault(node, dict())
            if node == self.node_id:
                node_attributes[node]['act'] = -1
            else:
                node_attributes[node]['act'] = one_hot_act[i]

        nx.set_node_attributes(new_G, node_attributes)
        self.G = new_G  # 更新图


    def to_Data(self):
        """
        将数据转化为Data
        :return:
        """
        # 1.获取节点ID
        node_ids = self.node_ids
        # 2.获取节点属性x
        node_attributes = dict(self.G.nodes(data=True))  # 获取图的属性字典
        attr_name_list = list(node_attributes[node_ids[0]].keys())  # 这里的0是必要的，表示索引0号节点，可能没有节点，遇到了后面处理
        for attr_name in attr_name_list:
            attrs = nx.get_node_attributes(self.G,attr_name).values()  # 取相应属性的所有值
            if 'x' not in locals():
                 x = [list(attrs)]
            else:
                x = np.concatenate((x,[list(attrs)]),axis=0)
        # print(x.T)
        # 3.获取邻接矩阵
        adjacency = torch.zeros([self.node_num,self.node_num])
        index_where = np.array(self.G.nodes)
        for src,std in self.G.edges():
            adjacency[index_where == src, index_where == std] = 1
            adjacency[index_where == std, index_where == src] = 1
        # adjacency = np.concatenate(([src_list],[std_list]),axis=0)
        # 4.将邻接矩阵正则化
        norm_adjacency = adjacency / torch.sum(adjacency,dim=1,keepdim=True)

        return self.Data(features=torch.tensor(x.T,dtype=torch.float), adjacency=norm_adjacency)  # 需要注意的是这里输出是张量


    # 获取邻居节点
    def get_neighbors(self):
        """
        获取邻居节点和邻居节点热索引
        :return:
        """
        assert self.node_id, "请先进行观测"
        node = self.node_id
        # print('当前节点',node)
        neighbors = list(self.G.neighbors(node))
        neighbor_hot = np.zeros(self.G.number_of_nodes(),dtype=bool)
        for index, node in enumerate(self.node_ids):
            if node in neighbors:
                neighbor_hot[index] = True
        # print(neighbor_hot)
        return neighbors,neighbor_hot


    def __getstate__(self):
        return self.G


    def __setstate__(self, G):
        self.G = G
        self.node_ids = np.array(self.G.nodes)
        self.node_num = len(self.G.nodes)
        self.Data = setting.Data


if __name__ == '__main__':
    G = nx.read_gpickle('graph.pkl')
    state = GetState(G)
    with open('attribute_dict.pkl','rb') as file:
        node_attr_dict = pickle.load(file)
    state.up_global(node_attr_dict)
    state.up_local(src=1)
    a = np.zeros(G.number_of_nodes(),dtype=np.int)
    a[2] = 1
    state.up_act(a)
    state_data = state.to_Data()
    neighbors, neighbor_hot = state.get_neighbors()
    print('\n属性矩阵\n',state_data.x)
    print('\n邻接矩阵\n',state_data.adjacency)
    print('\n当前节点的邻居节点\n',neighbors)
    print('\n当前节点的邻居节点的独热编码\n',neighbor_hot)

    with open('state_sample.pkl','wb') as f:
        pickle.dump(state,f)

    with open('state_sample.pkl','rb') as f:
        save_state = pickle.load(f)

    data = save_state.to_Data()
    print(data.x)
    print(data.adjacency)



    # DataT = DataTrans(G)
    # gdata = DataT.to_Data()
    # print(gdata.node_id)
    # node = 14
    # neighbors,neighbor_hot = DataT.get_neighbors(node)
    # print(neighbors)
    # print(neighbor_hot)















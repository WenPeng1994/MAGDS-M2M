#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Auther:Evans_WP
@Contact:WenPeng19943251019@163.com
@Time:2024/3/2 21:19
@File:setting.py
@Desc:用于设置参数
"""
# 流量矩阵
import numpy as np
# import algorithms
from collections import namedtuple
import copy

traffic_matrix_1 = np.diag([15,270,15])

traffic_matrix_2 = np.matrix([[0,0,0],[10,10,10],[0,0,0]])

traffic_matrix_3 = np.matrix([[10,0,0],[10,0,0],[10,0,0]])

traffic_matrix_4 = np.matrix([[40,30,30],[30,40,30],[30,30,40]])

traffic_matrix_5 = np.matrix([[20,15,15],[15,20,15],[15,15,20]])

traffic_matrix_6 = np.matrix([[10,0,0],[0,0,0],[0,0,0]])

traffic_matrix = traffic_matrix_4

# 能量使用参数
transmission_energy = 10  # mW  传输功率
receiving_energy = 3  # mW 接收功率
idle_energy = 0.1  # mW  模型待机功率


# 关键节点选择参数
proportion = 0.3

# 使用的算法
# use_algorithms = algorithms.random_short_path_hop

# 数据包总数
MAX_TASK = int(np.sum(traffic_matrix))
pre_state = copy.deepcopy(MAX_TASK)  # 用于记录之前的节点上的数据包个数，是一个可变的变量


# 传输数据包结构
packet_str = ['src', 'std', 'index', 'precursor']

# 目标函数的权衡系数
beta = 0.001

# 构造模型的输入数据
Data = namedtuple('Data', ['features', 'adjacency'])

# 终端奖励
terminal_reward = 2000

# 单位奖励
unit_reward = terminal_reward**(1/MAX_TASK)

# 避免除数出现零的参数
epsilon = 1e-5

# 设置不同的奖励函数,
# 1.包含路径，数据包和累积使用能量三个部分
# 2.仅仅包含数据和累积使用能量两个部分,比例为0.5:0.5，学习慢，陷入局部最优，增大学习率后有明显效果。问题是学习一段时间后，它会跳出最优解
#  而且学习不回来了？会不会它追求了其他两个指标的优化。
# 3.仅仅包含数据包总数
reward_type = 6


# 学习率调整参数
adjust_learning_rate_episodes = 5   # 学习率调整步数
adjust_learning_rate_range = 0.98    # 学习率调整幅度
adjust_learning_rate_limit = 0.0001  # 学习率调整的极限，最多变为原来的千分之一,以actor_net为主。

# 探索率
eps = 0.1


# 回路惩罚
loop_punishment = -1

# 保存间隔
save_episode = 5

# 算法1权衡系数
beta_no = 0.01

# topo路径
xml_path = './topo1.xml'
path_proof = './topo1.xml'


# 动作修正
action_modification = True
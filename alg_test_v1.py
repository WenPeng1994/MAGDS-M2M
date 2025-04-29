#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Auther:Evans_WP
@Contact:WenPeng19943251019@163.com
@Time:2024/4/5 21:14
@File:main.py
@Desc:****************
"""
import torch

import algorithms
import setting
from algorithms import DDPG, MADDPG
import pickle
from MARL_env import PacketRoutingEnv,Topo
import numpy as np
import logging
from comments.utils import log_management,ReplayBuffer,Evaluate
import matplotlib.pyplot as plt
import time
import random
import os
# 全局设置随机种子
# seed = 1234
seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from scipy.io import savemat

import statistics

# TODO:  绘制图片并保存数据
def convergence_result_display(evaluate_value):
    formatted_time = time.strftime("%Y-%m-%d_%H:%M", time.localtime())
    with open('./result/data/evaluate_{}.pkl'.format(formatted_time), 'wb') as f:
        pickle.dump(evaluate_value, f)
    num_list = []  # 迭代代数
    for episode_num, _ in enumerate(evaluate_value):
        num_list.append(episode_num)

    plt.plot(num_list, evaluate_value)
    plt.title('evaluate of agents')
    plt.xlabel('episodes')
    plt.ylabel('evaluate')
    plt.legend()  # 图例可以进一步设置
    plt.savefig('./result/figure/evaluate_of_agents_{}.png'.format(formatted_time))

    plt.show()


# 绘制节点数据包变化曲线
def node_package_display(node_package_nums_dict,model_name,display=True):
    value = []  # 格式化，避免报错
    for key,value in node_package_nums_dict.items():
        msg_num_list = value
        slot_time = range(len(value))
        data_dict = {f'{model_name}_slot_time':slot_time,f'{model_name}_msg_num':msg_num_list}
        savemat(f'./result/data/{model_name}_node{key}_msg_number.mat',data_dict)

        if display:
            plt.figure(key)
            # 绘制图
            plt.plot(slot_time,msg_num_list)
            plt.xlabel('slot_time')
            plt.ylabel('msg_num')
            plt.title('{} msg number of node {}'.format(model_name,key))
            plt.show()
    else:
        print(f'总用时{len(value)}slot')
    return len(value)

# 绘制节点能量使用量
def node_energy_display(node_energy_use_dict,model_name,display=True):
    node_info_list = []
    node_energy_list = []
    for key,value in node_energy_use_dict.items():
        node_info_list.append(str(key))
        node_energy_list.append(value)

    data_dict = {f'{model_name}_node_info': node_info_list, f'{model_name}_node_energy': node_energy_list}
    savemat(f'./result/data/{model_name}_node_energy.mat', data_dict)

    if display:
        plt.figure(figsize=(8, 6))
        plt.barh(node_info_list, node_energy_list)
        for i, num in enumerate(node_energy_list):
            plt.text(num, i, str(num), ha='left', va='center')
        plt.xlabel('used energy of node')
        plt.ylabel('node')
        plt.title(f'{model_name} Node energy usage statistics')
        plt.tight_layout()  # 自动调整子图参数
        plt.show()
    print(f'节点上使用能量的方差为:{statistics.stdev(node_energy_list)}')
    return statistics.stdev(node_energy_list)

# 绘制链路上数据通过量
def link_msg_display(link_msg_dict,model_name,display=True):
    link_info_list = []
    link_msg_list = []
    for key,value in link_msg_dict.items():
        if value > 0:  # 去除链路上通过数据为零结果
            link_info_list.append(str(key))
            link_msg_list.append(value)

    data_dict = {f'{model_name}_link_info': link_info_list, f'{model_name}_link_msg': link_msg_list}
    savemat(f'./result/data/{model_name}_links_msg_number.mat', data_dict)


    if display:
        plt.figure(figsize=(8, 6))
        plt.barh(link_info_list, link_msg_list)
        for i, num in enumerate(link_msg_list):
            plt.text(num, i, str(num), ha='left', va='center')
        plt.xlabel('msg number of link')
        plt.ylabel('link')
        plt.title(f'{model_name} Information statistics on links')
        plt.tight_layout()  # 自动调整子图参数
        plt.show()

def model_evaluate_display(episodes_list,time_statistic_list,var_energy_list,model_ver,display=True):
    data_dict = {f'{model_ver}_episodes_list':episodes_list,f'model_evaluate':[time_statistic_list,var_energy_list]}
    savemat(f'./result/data/{model_ver}_model_evaluate.mat', data_dict)
    if display:
        plt.figure()
        plt.plot(episodes_list, time_statistic_list)
        plt.xlabel('episode')
        plt.ylabel('slot_time')
        plt.title('time statistic of {}_{}'.format(model_ver,episodes_list[-1]+setting.save_episode))
        plt.savefig('time statistic of {}_{}'.format(model_ver,episodes_list[-1]+setting.save_episode))
        plt.show()
        plt.figure()
        plt.plot(episodes_list, var_energy_list)
        plt.xlabel('episode')
        plt.ylabel('energy_var')
        plt.title('energy variance of {}_{}'.format(model_ver,episodes_list[-1]+setting.save_episode))
        plt.savefig('energy variance of {}_{}'.format(model_ver,episodes_list[-1]+setting.save_episode))
        plt.show()


# 模型演化评价
def model_evaluate(env,maddpg,model_ver,model_eps,test_limit_length):
    # 1.读取模型参数
    for i, agt in enumerate(maddpg.agents):
        # 构造模型的路劲
        model_path = './result/model/' + 'main_' + model_ver + '_actor' + model_eps + '-' + str(i) + '.pth'
        model_parameters = torch.load(model_path)
        agt.actor.load_state_dict(model_parameters)

    done = False
    step = 0
    # 2. 提取数据
    node_package_nums_dict = {}  # 节点数据列表
    node_package_num_dict = {}  # 形式化，避免报错
    node_energy_use_dict = {}  # 形式化，避免报错
    link_msg_dict = {}  # 形式化，避免报错

    for node in env.G.nodes:
        node_package_nums_dict.setdefault(node, list())

    while not done and step < test_limit_length:
        node_package_num_dict, node_energy_use_dict, link_msg_dict, done = env.MARL_test_step(maddpg.agents,
                                                                                                  explore=False)
        for node in env.G.nodes:
            node_package_nums_dict[node].append(node_package_num_dict[node])
        step += 1

    # 3.绘制图
    model_name = f'main_{model_ver}_{model_eps}_episode'
    total_time = node_package_display(node_package_nums_dict, model_name)
    var_energy = node_energy_display(node_energy_use_dict, model_name)
    return total_time,var_energy



if __name__ == '__main__':
    # 1.将文件名写入文件名
    file_name = os.path.basename(__file__).split('.')[0]
    formatted_time = time.strftime("%Y-%m-%d_%H:%M", time.localtime())
    logger = log_management(file_name)
    logger.info(f'\n\n{formatted_time}{file_name}运行结果\n\n')

    ### main_v11.py后将所有参数全部放在main中
    setting.traffic_matrix = setting.traffic_matrix_4   # 流量矩阵
    setting.reward_type = 6                             # 奖励类型
    setting.eps = 0.1                                   # 探索率
    setting.xml_path = './topo1.xml'
    setting.path_proof = './topo1.xml'


    topo = Topo(xml_path=setting.xml_path,traffic_matrix=setting.traffic_matrix_4)
    logger.info(f'\n全局参数\n流量矩阵:\n{setting.traffic_matrix}'
          f'\n奖励类型:\n{setting.reward_type}'
          f'\n探索率:\n{setting.eps}'
          f'\n节点序列：\n{topo.G.nodes}'
          f'\n数据源节点序列：\n{topo.src_node_set}'
          f'\n数据源节点序列：\n{topo.std_node_set}'
          )
    ########

    agents_node = [6, 8, 9, 10, 15]
    agents_num = len(agents_node)
    state_dim = 4
    critic_input_dim = (state_dim + 1 - 2) * agents_num + 2  # 在评价网络输入中去掉重复的量
    critic_hidden_dim = agents_num


    ### 超参数设置
    actor_lr = 0.002           # 动作网络学习率
    critic_lr = 0.00002        # 评价网络学习率
    gamma = 0.95               # 折扣因子
    tau = 1e-2                 # 软学习因子
    num_episodes = 1500         # 序列数
    episode_length = 500       # 每条序列的最大长度
    eva_length = 500           # 测试序列长度
    eva_episode = 1            # 评价周期数
    min_save_episode = 100     # 最小保存网络代数
    setting.save_episode = 5           # 保存间隔

    ###### 部署参数

    alg_name = "magddpg"       # 多智能体图强化学习
    # alg_name = 'other_alg'
    model_evaluate_type = True  # 控制是否进行模型演化
    # model_evaluate_type = False  # 控制是否进行模型演化
    model_ver = "v132_7_5"          # 模型版本
    model_eps = "1380"          # 模型代数


    test_limit_length = 500    # 测试极限步数
    algorithm = algorithms.avoid_loop_short_path

    setting.adjust_learning_rate_limit = actor_lr*0.001

    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = PacketRoutingEnv(agents_node)
    maddpg = MADDPG(env, device, actor_lr, critic_lr, state_dim, critic_input_dim, critic_hidden_dim, gamma, tau)





    if model_evaluate_type and alg_name == "magddpg":
        time_statistic_list = []
        var_energy_list = []
        episodes_list = []
        for model_eps_num in range(0,num_episodes,setting.save_episode):
            print(f'第{model_eps_num}代模型')
            model_eps = str(model_eps_num)
            model_exist = True
            for i, agt in enumerate(maddpg.agents):
                # 构造模型的路劲
                try:
                    model_path = './result/model/' + 'main_' + model_ver + '_actor' + model_eps + '-' + str(i) + '.pth'
                    model_parameters = torch.load(model_path)
                    agt.actor.load_state_dict(model_parameters)
                except:
                    model_exist = False
                    break
            if model_exist:
                env.reset()
                done = False
                step = 0
                # 2. 提取数据
                node_package_nums_dict = {}  # 节点数据列表
                node_package_num_dict = {}  # 形式化，避免报错
                node_energy_use_dict = {}  # 形式化，避免报错
                link_msg_dict = {}  # 形式化，避免报错

                for node in env.G.nodes:
                    node_package_nums_dict.setdefault(node, list())

                while not done and step < test_limit_length:
                    if alg_name == "magddpg":
                        node_package_num_dict, node_energy_use_dict, link_msg_dict, done = env.MARL_test_step(
                            maddpg.agents,
                            explore=False)
                    else:

                        node_package_num_dict, node_energy_use_dict, link_msg_dict, done = env.other_alg_step(
                            algorithm)
                    for node in env.G.nodes:
                        node_package_nums_dict[node].append(node_package_num_dict[node])
                    step += 1
                # 3.绘制图
                if alg_name == "magddpg":
                    model_name = f'main_{model_ver}_{model_eps}_episode'
                else:
                    model_name = f'main_{alg_name}_episode'
                time_statistic = node_package_display(node_package_nums_dict, model_name,display=False)
                var_energy = node_energy_display(node_energy_use_dict, model_name,display=False)
                time_statistic_list.append(time_statistic)
                var_energy_list.append(var_energy)
                episodes_list.append(model_eps_num)
            else:
                continue
        model_evaluate_display(episodes_list,time_statistic_list,var_energy_list,model_ver)
    else:
        # 1.读取模型参数
        for i,agt in enumerate(maddpg.agents):
            # 构造模型的路劲
            model_path = './result/model/' + 'main_' + model_ver + '_actor' + model_eps + '-' + str(i) + '.pth'
            model_parameters = torch.load(model_path)
            agt.actor.load_state_dict(model_parameters)

        done = False
        step = 0
        # 2. 提取数据
        node_package_nums_dict = {}  # 节点数据列表
        node_package_num_dict = {}   # 形式化，避免报错
        node_energy_use_dict = {}  # 形式化，避免报错
        link_msg_dict = {}  # 形式化，避免报错

        for node in env.G.nodes:
            node_package_nums_dict.setdefault(node,list())


        while not done and step < test_limit_length:
            if alg_name == "magddpg":
                node_package_num_dict, node_energy_use_dict, link_msg_dict, done = env.MARL_test_step(maddpg.agents,
                                                                                     explore=False)
            else:
                node_package_num_dict, node_energy_use_dict, link_msg_dict, done = env.other_alg_step(algorithm)
            for node in env.G.nodes:
                node_package_nums_dict[node].append(node_package_num_dict[node])
            step += 1

        # 3.绘制图
        if alg_name == "magddpg":
            model_name = f'main_{model_ver}_{model_eps}_episode'
        else:
            model_name = f'main_{alg_name}_episode'
        node_package_display(node_package_nums_dict,model_name)
        node_energy_display(node_energy_use_dict,model_name)
        link_msg_display(link_msg_dict,model_name)


# v132_5_6_model_evaluate.mat
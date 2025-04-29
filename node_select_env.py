#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Auther:Evans_WP
@Contact:WenPeng19943251019@163.com
@Time:2024/5/18 11:27
@File:node_select_env.py
@Desc:节点重要性选择环境
"""
import random
import numpy as np
import copy
import time
import matplotlib.pyplot as plt
import pickle
from scipy.io import savemat

class Node_importance_env:
    def __init__(self,G,n):
        from MARL_env import PacketRoutingEnv
        import setting
        agents_node = list(G.nodes)
        self.env = PacketRoutingEnv(agents_node)  # 为了使用之前的代码，初始化需要智能体节点，实际上没有作用，这里直接给所有节点
        self.G = G  # 用于保存topo，包含了邻接矩阵
        self.n = n  # 最多重要性节点的个数
        self.M = setting.traffic_matrix  # 用于保存任务矩阵，规格应该是n×n，用于计算奖励
        self.nodes = list(self.G.nodes)  # 节点列表
        self.importance_nodes = set()  # 用于保存重要性节点的集合
        self.not_importance_nodes = set(self.nodes)  # 用于保存非重要性节点的集合
        self.state = [self.importance_nodes,self.not_importance_nodes]   # 当前状态

    def step(self,action):  # 外部调用这个函数来改变重要性节点集合,action的数据类型用list
        # cur_state_reward = self.reward_calculation(self.state)
        # print(f"当前状态{self.state}的奖励{cur_state_reward}")
        state = copy.deepcopy(self.state)
        select_node = self.nodes[int(np.flatnonzero(action))%len(self.nodes)]  # np.flatnonzero(),numpy中非零元素的线性索引
        if select_node in self.importance_nodes:  # 如果节点在重要性节点中，执行切换
            self.importance_nodes.discard(select_node)
            self.not_importance_nodes.add(select_node)
        else:
            self.not_importance_nodes.discard(select_node)
            self.importance_nodes.add(select_node)
        next_state = self.state
        reward = self.reward_calculation(next_state,select_node)
        # if select_node in self.importance_nodes and reward == 0:  # 非重要节点变重要节点，但没奖励
        #     reward -= 1  # 给予惩罚
        # elif select_node in self.not_importance_nodes and reward == 0: # 重要节点变非重要节点,没损失
        #     reward += 1  # 给予奖励
        # next_state_reward = self.reward_calculation(next_state)
        # print(f"下一个状态{next_state}的奖励{next_state_reward}")
        print(f"下一个状态{next_state}")
        # reward = (next_state_reward-cur_state_reward)
        print(f"转移奖励{reward}")
        done = False
        # if next_state[0] == {1, 8, 10, 12, 14, 9, 11, 4, 5, 7}:
        #     print(next_state)
        if len(next_state[0]) == self.n:
            done = True

        # self.state = next_state  # 更新状态,已经更新了，不用重复更新
        return state,reward,next_state,done


    def reward_calculation(self,state,select_node):
        """
        用于计算状态间奖励：
        在重要性节点上使用随机简单路径的方法，在非重要性节点上使用随机最短路径的方法。
        :param state: 状态
        :return:
        """
        self.env.reset()  # 每次跑前将环境进行初始化
        state_randomness = 0
        done = False
        while not done:
            randomness,done = self.env.node_importance_epsilon_step(state,select_node,setting.beta_no)
            # 使用epsilon贪心策略
            state_randomness += randomness

        return state_randomness

    def test_reward_calculation(self,state):
        """
        用于计算某状态下的奖励
        :param state:
        :return:
        """
        self.env.reset()
        state_randomness = 0
        done = False
        while not done:
            randomness, done = self.env.state_importance_step(state)
            state_randomness += randomness

        return state_randomness


    def reset(self):  # 回到初始状态，重要性节点集合为空
        self.importance_nodes = set()
        self.not_importance_nodes = set(self.G.nodes)
        self.state = [self.importance_nodes, self.not_importance_nodes]  # 状态初始化
        return self.state


class QLearning:
    """ Q-learning算法 """
    def __init__(self,G,epsilon,alpha,gamma):
        self.G = G
        self.nodes = self.G.nodes()
        self.node_num = len(self.G.nodes())
        self.action_class = 2
        self.action_space = self.action_space_gen(self.action_class)
        action_num = self.action_space.shape[0]
        self.node_choice = 2
        self.states_space = self.states_space_gen(self.node_choice)
        state_num = self.states_space.shape[0]
        self.Q_table = np.zeros([state_num,action_num])  # Q(s,a)表示
        self.epsilon = epsilon  # 探索率
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣率

    # 生成状态空间
    def states_space_gen(self,node_choices):
        node_num = len(self.nodes)  # 节点数
        states_space = np.array([])
        v_num = 0
        if node_num == 0:
            states_space = np.array([])
        else:
            for node_index in range(node_num):
                if node_index == 0:
                    states_space = np.array(range(node_choices)).T
                    v_num = copy.deepcopy(node_choices)
                else:
                    for index,node_choice in enumerate(range(node_choices)):
                        if index == 0:
                            states_space0 = np.vstack((states_space,np.array([node_choice]*v_num).T))
                        else:
                            states_space = np.hstack((states_space0,np.vstack((states_space,np.array([node_choice]*v_num).T))))
                    v_num *= node_choices
        return states_space.T


    # 生成动作空间
    def action_space_gen(self,action_class):
        action_num = self.node_num*action_class
        action_space = np.array([])
        for action_index in range(action_num):
            action0 = np.zeros(action_num)
            action0[action_index] = 1
            if action_index == 0:
                action_space = action0
            else:
                action_space = np.vstack((action_space,action0))
        return action_space


    # 选取下一步动作
    def take_action(self,state,exploration=True):
        explor = False
        import_node_num = state[0]
        action = np.zeros(self.node_num*self.action_class)
        node_list = list(self.nodes)
        # print(f"当前的探索率{self.epsilon}")
        if np.random.random() < self.epsilon and exploration:  # 二者不满足其一的情况下，就用贪心算法
            point_choice = np.random.choice(list(range(self.node_num)))
            if node_list[point_choice] in import_node_num:
                action[point_choice+self.node_num] = 1  # 在重要节点中删掉该节点
            else:
                action[point_choice] = 1  # 将该节点放入重要节点中
            explor = True
        else:
            max_value = np.max(self.Q_table[self.state_index(state)])
            points = np.where(self.Q_table[self.state_index(state)] == max_value)[1]
            point_choice = np.random.choice(points)
            if point_choice % len(node_list) in import_node_num:  # 节点在重要性节点中
                action[point_choice] = 1
            else:  # 节点不在重要性节点中
                action[point_choice % len(node_list)] = 1
        return action,explor


    # 索引状态所在的位置
    def state_index(self,state):
        # 注意这里的状态是集合向量的形式
        importance_node_set = state[0]
        logtic = np.zeros(self.node_num)
        for node_index,node in enumerate(self.nodes):
            if node in importance_node_set:
                logtic[node_index] = 1
        return np.where((self.states_space == logtic).all(axis=1))[0]


    # 索引动作所在的位置
    def action_index(self,action):
        return np.where((self.action_space == action).all(axis=1))[0]

    # 更新
    def update(self,s0,a0,r,s1):
        td_error = r + self.gamma * self.Q_table[self.state_index(s1)].max() \
                   - self.Q_table[self.state_index(s0),self.action_index(a0)]
        self.Q_table[self.state_index(s0),self.action_index(a0)] += self.alpha * td_error

# TODO:  绘制决策数图片并保存数据
def node_decision_number_display(acc_decision_list,explor_episode,import_node_num):
    formatted_time = time.strftime("%Y-%m-%d_%H:%M", time.localtime())
    with open('./result/data/acc_decision_{}.pkl'.format(formatted_time), 'wb') as f:
        pickle.dump(acc_decision_list, f)
    num_list = []  # 迭代代数
    for episode_num, _ in enumerate(acc_decision_list):
        num_list.append(episode_num)
    acc_epis = []
    for epis in explor_episode:
        acc_epis.append(acc_decision_list[epis])

    plt.plot(num_list, acc_decision_list)
    plt.scatter(explor_episode,acc_epis,c='r',marker='*')
    data_dict = {f'node_import_num_list_{import_node_num}': num_list, 'acc_decision_list':acc_decision_list,
                 f'explor_episode_{import_node_num}':explor_episode,'acc_epis':acc_epis}
    savemat(f'./result/data/node_import_{import_node_num}_data.mat', data_dict)
    plt.title('Cumulative decision number')
    plt.xlabel('episodes')
    plt.ylabel('cumulative decision number')
    # plt.legend()  # 图例可以进一步设置
    plt.savefig('./result/figure/cumulative_decision_number_{}.png'.format(formatted_time))
    plt.show()

# 评价函数
def evaluation_function(alg,node_importance_env,test_num):
    accum_reward_list = []
    for episode in range(test_num):
        state = node_importance_env.reset()
        done = False
        while not done:
            action, _ = alg.take_action(state,exploration=False)
            state, reward, next_state, done = node_importance_env.step(action)
            state = next_state
        accum_reward_list.append(node_importance_env.test_reward_calculation(state))
        print(f'测试周期{episode}下的重要性节点{state[0]}')
        # 计算状态下的奖励
    print(f"10次测试平均奖励函数值为{np.mean(accum_reward_list)}")
    return np.mean(accum_reward_list)



if __name__ == '__main__':
    random.seed(1234)
    np.random.seed(1234)
    from MARL_env import Topo
    import setting
    from MARL_env import PacketRoutingEnv
    mission = setting.traffic_matrix
    topo = Topo(traffic_matrix=mission)
    import_node_num = 10

    # print(topo.G.nodes)
    # node_num = len(topo.G.nodes)
    # for node_index in range(node_num):
    #     action_point = node_index
    #     action = np.zeros(node_num)  # 动作用tensor
    #     action[action_point] = 1
    #
    #     node_importance_env = Node_importance_env(topo.G,import_node_num)
    #     next_state,reward,done = node_importance_env.step(action)
    #     print(next_state,reward,done)

    epsilon = 0.99    # 探索率
    alpha = 0.01  # 学习率
    gamma = 0.9      # 折扣率
    # 探索率衰减
    reject_ratio = 0.5  # 衰减概率
    reject_steps = 50   # 衰减步数

    node_importance_env = Node_importance_env(topo.G,import_node_num)
    alg = QLearning(topo.G,epsilon,alpha,gamma)

    # state_space = alg.states_space_gen(2)
    # print(np.where((state_space==[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).all(axis=1))[0])

    # # 动作编号的测试
    # action_point = 1
    # action = np.zeros(node_num)  # 动作用tensor
    # action[action_point] = 1
    # print(alg.action_index(action))
    #
    # # 状态编号的测试
    # state_test = [set(range(14)),set([14])]
    # print(alg.state_index(state_test))

    episode_num = 600

    accum_reward_list = []
    explor_episode = []
    for episode in range(episode_num):
        # 循环次数
        if episode % reject_steps == 0 and episode > 0:
            alg.epsilon *= reject_ratio
        episode_reward = 0
        state = node_importance_env.reset()
        done = False
        explor0 = False
        while not done:
            action, explor = alg.take_action(state)
            if explor:
                explor0 = True
            print(f"状态变化{state}")
            state,reward,next_state,done = node_importance_env.step(action)
            episode_reward += reward
            alg.update(state,action,reward,next_state)
            # print(state)
            state = next_state
        if explor0:
            explor_episode.append(episode)
        print(f"{episode}的奖励值{episode_reward}")
        accum_reward_list.append(episode_reward)
    print(f"所有周期的奖励值{accum_reward_list}")
    node_decision_number_display(accum_reward_list,explor_episode,import_node_num)
    print(f"探索发生的周期{explor_episode}")

    # 测试次数
    test_num = 10
    evaluation_function(alg,node_importance_env,test_num)















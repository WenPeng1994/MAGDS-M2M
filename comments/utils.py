#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Auther:Evans_WP
@Contact:WenPeng19943251019@163.com
@Time:2024/3/29 21:36
@File:utils.py
@Desc:用于存储常用的功能性函数
"""
import random
from setting import packet_str
import torch
import torch.nn.functional as F
import numpy as np
import logging
import time
import collections
import setting
import pickle
import matplotlib.pyplot as plt
from scipy.io import savemat


# 随机概率模拟
def random_simulation(rate):
    """
    随机概率模拟
    :param rate: 丢包率
    :return:
    """
    # print(setting.xml_path)
    # if rate == 0.8:
    #     print(rate)
    if random.random() < 1 - rate:  # 1-rate 传输成功
        return True  # 传输成功
    else:
        return False  # 传输失败


# 封装leftpop()
def re_popleft(buffer):
    try:
        msg = buffer.popleft()
    except IndexError:
        msg = [None for _ in packet_str]
    return msg


def gumbel_softmax(logits, temperature=1.0):
    """ 从Gumbel-Softmax分布中采样，并进行离散化 """
    y = gumbel_softmax_sample(logits, temperature)
    y_hard = onehot_from_logits(y)  # epsilon-贪心策略
    y = (y_hard.to(logits.device) - y).detach() + y
    # 返回一个y_hard的独热量，但是它的梯度是y的梯度，我们既能够得到一个与环境交互的离散动作，又可以正确地反向传递
    return y


def gumbel_softmax_sample(logits, temperature):
    """ 从Gumbel-Softmax分布中采样 """
    y = logits + sample_gambel(logits.shape, tens_type=type(logits.data)).to(logits.device)*1e-1
    return F.softmax(y / temperature, dim=1)


def sample_gambel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """ 从Gumbel(0,1)分布中采样 """
    U = torch.autograd.Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)


def onehot_from_logits(logits, eps=setting.eps):
    """ 生成随机动作，转换成独热(one-hot)形式 """
    while True:
        epss_list = list(range(logits.shape[1]))
        random.shuffle(epss_list)
        epss = torch.tensor(epss_list, requires_grad=False, device=logits.device) * 1e-5
        # print(epss)
        # 增加一个小数，用于区分一样的动作
        logits0 = logits + epss
        # print(logits0)
        argmax_acs = (logits0 == logits0.max(1, keepdim=True)[0]).float()
        action = argmax_acs.detach().cpu().view(-1).numpy()
        if sum(action) == 1:
            break

    # 生成随机动作，转换成独热形式
    rand_acs = torch.autograd.Variable(torch.eye(logits.shape[1])[[np.random.choice(
        range(logits.shape[1]), size=logits.shape[0]
    )]], requires_grad=False).to(logits.device)
    # 通过epsilon-贪婪算法来选择用哪个动作
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                        enumerate(torch.rand(logits.shape[0]))])


# 基于batch的gumbel_softmax函数
def gumbel_softmax_batch(logits, temperature=1.0):
    """ 从Gumbel-Softmax分布中采样，并进行离散化 """
    y = gumbel_softmax_sample_batch(logits, temperature)
    y_hard = onehot_from_logits_batch(y)  # epsilon-贪心策略
    y = (y_hard.to(logits.device) - y).detach() + y
    # 返回一个y_hard的独热量，但是它的梯度是y的梯度，我们既能够得到一个与环境交互的离散动作，又可以正确地反向传递
    return y




def gumbel_softmax_sample_batch(logits, temperature):
    """ 从Gumbel-Softmax分布中采样 """
    y = logits + sample_gambel_batch(logits.shape, tens_type=type(logits.data)).to(logits.device)
    return F.softmax(y / temperature, dim=1)


def sample_gambel_batch(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """ 从Gumbel(0,1)分布中采样 """
    U = torch.autograd.Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)


def onehot_from_logits_batch(logits, eps=setting.eps):
    """ 生成随机动作，转换成独热(one-hot)形式 """
    # epss = torch.tensor(list(range(logits.shape[1])), requires_grad=False, device=logits.device) * 1e-5
    # print(epss)
    # 增加一个小数，用于区分一样的动作
    # logits = logits + epss
    # argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    while True:
        epss_list = list(range(logits.shape[1]))
        random.shuffle(epss_list)
        epss = torch.tensor(epss_list, requires_grad=False, device=logits.device) * 1e-5
        # print(epss)
        # 增加一个小数，用于区分一样的动作
        logits0 = logits + epss
        argmax_acs = (logits0 == logits0.max(1, keepdim=True)[0]).float()
        action = argmax_acs.detach().cpu().view(-1).numpy()
        if sum(action) == 1:
            break
    # 生成随机动作，转换成独热形式
    rand_acs = torch.autograd.Variable(torch.eye(logits.shape[1])[[np.random.choice(
        range(logits.shape[1]), size=logits.shape[0]
    )]], requires_grad=False).to(logits.device)
    # 通过epsilon-贪婪算法来选择用哪个动作
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                        enumerate(torch.rand(logits.shape[0]))])


def onehot_from_logits_max(logits, eps=0.01):
    """ 生成随机动作，转换成独热(one-hot)形式 """
    # epss = torch.tensor(list(range(logits.shape[1])), requires_grad=False, device=logits.device) * 1e-5
    # print(epss)
    # 增加一个小数，用于区分一样的动作
    # logits = logits + epss
    while True:
        epss_list = list(range(logits.shape[1]))
        random.shuffle(epss_list)
        epss = torch.tensor(epss_list, requires_grad=False, device=logits.device) * 1e-5
        # epss = torch.tensor(random.shuffle(list(range(logits.shape[1]))), requires_grad=False, device=logits.device) * 1e-5
        # print(epss)
        # 增加一个小数，用于区分一样的动作
        logits0 = logits + epss
        argmax_acs = (logits0 == logits0.max(1, keepdim=True)[0]).float()
        action = argmax_acs.detach().cpu().view(-1).numpy()
        if sum(action) == 1:
            break
    # argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    # 生成随机动作，转换成独热形式
    return argmax_acs


# 日志管理
def log_management(file_name):
    formatted_time = time.strftime("%Y-%m-%d_%H:%M", time.localtime())
    log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S %p")
    # date_format = "%m/%d/%Y %H:%M:%S %p"
    # logging.basicConfig(filename='./logging/run_{}.log'.format(formatted_time), level=logging.INFO, format=log_format,
    #                     datefmt=date_format)
    # 创建logger
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.INFO)
    # 写入文件file_handler
    file_handler = logging.FileHandler('./logging/{}_{}.log'.format(file_name,formatted_time))
    file_handler.setLevel(logging.INFO)
    # 创建一个handler，用于输出日志到控制台
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.INFO)

    # 设置formatter
    file_handler.setFormatter(log_format)
    # console_handler.setFormatter(log_format)

    # 将handler添加到logger
    logger.addHandler(file_handler)
    # logger.addHandler(console_handler)

    return logger


# 经验缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 设置缓冲区

    def add(self, state, action, reward, next_state, done, precursor):
        self.buffer.append((state, action, reward, next_state, done,precursor))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, precursor = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done, precursor

    def size(self):
        return len(self.buffer)

    def save(self,filename):   # 保存
        with open(filename,'wb') as f:
            pickle.dump(self.buffer,f)

    def load(self,filename):   # 读取
        with open(filename,'rb') as f:
            self.buffer = pickle.load(f)

# 评价类
class Evaluate:
    def __init__(self,eva_episode,eva_length):
        """
        参数初始化
        :param adjust_lr_start_step: 调整学习率的起始步数
        :param eva_episode: 评价周期数
        :param eva_length: 每个评价周期的评价长度
        """
        self.adjust_lr_start_step = 0
        self.eva_episode = eva_episode
        self.eva_length = eva_length

    def evaluate(self,env,maddpg,i_episode):
        """
        评价函数
        :param env: 环境
        :param maddpg: 模型
        :peram i_episode: 当前步
        :return:
        """
        returns = np.zeros(len(env.agents_node))
        for _ in range(self.eva_episode):
            env.reset()
            for t_i in range(self.eva_length):
                agents = maddpg.agents
                _, acts, rews, _, done, _ = env.avoide_loop_step(agents, explore=False)
                # TODO:20240621修改
                # _, acts, rews, _, done = env.step(agents, explore=False)
                rews = np.array(rews)
                returns += rews / self.eva_episode
                if done:
                    print('到达终点')
                    if not self.adjust_lr_start_step:
                        self.adjust_lr_start_step = i_episode   # 记录评价起始位置
                    break


            # if not done and setting.reward_type != 7:
            #     # returns += setting.terminal_reward/setting.MAX_TASK*(setting.MAX_TASK-
            #     #             sum(env.node_package_num_dict.values()))/self.eva_episode
            #     # returns += setting.unit_reward*(setting.MAX_TASK-sum(env.node_package_num_dict.values()))/self.eva_episode
            #     pass
            if not done:
                returns += setting.unit_reward ** (
                        setting.MAX_TASK - sum(env.node_package_num_dict.values())) / self.eva_episode
            elif self.adjust_lr_start_step and (i_episode-self.adjust_lr_start_step) % setting.adjust_learning_rate_episodes == 0:
                # 只有在到达终点的情况下调整学习率，没有到达的状态不变
                for agt in maddpg.agents:
                    for actor_param_group,critic_param_group in zip(agt.actor_optimizer.param_groups,agt.critic_optimizer.param_groups):
                        # 显示调整前的学习率
                        if actor_param_group['lr'] > setting.adjust_learning_rate_limit:
                            print('第{}步调整前动作网络的学习率'.format(i_episode), actor_param_group['lr'])
                            print('第{}步调整前评价网络的学习率'.format(i_episode), critic_param_group['lr'])
                            actor_param_group['lr'] *= setting.adjust_learning_rate_range
                            critic_param_group['lr'] *= setting.adjust_learning_rate_range

                # for agt in maddpg.agents:
                #     for actor_param_group, critic_param_group in zip(agt.actor_optimizer.param_groups,
                #                                                      agt.critic_optimizer.param_groups):
                #         actor_param_group['lr'] *= setting.adjust_learning_rate_range
                #         critic_param_group['lr'] *= setting.adjust_learning_rate_range
        return returns.tolist()

# TODO：在utils.py中添加函数random_one_in_list，用于动作中具有多个1时，随机选择一个。
def random_one_in_list(lst):
    one_pos = [index for index, value in enumerate(lst) if value == 1]
    random_index, = random.sample(one_pos, 1)
    lst_new = [0] * len(lst)
    lst_new[random_index] = 1
    return lst_new


# TODO：20240623在utils中添加一个功能性函数tuple_t,用于将m×n的元组列表转化为n×m的元组列表
def tuple_t(tuple_list):
    # 将m×n的元组列表转化为n×m的元组列表
    # 1. 将元组列表转化为列表
    inter_var = [list(tuple_) for tuple_ in tuple_list]
    # 2. 将列表转化为转置后的元组列表
    return [tuple(inter_var[j][i] for j in range(len(inter_var))) for i in range(len(inter_var[0]))]


# TODO:  绘制图片并保存数据
def convergence_result_display(evaluate_value,file_name):
    formatted_time = time.strftime("%Y-%m-%d_%H:%M", time.localtime())
    with open('./result/data/{}_evaluate_{}.pkl'.format(file_name,formatted_time), 'wb') as f:
        pickle.dump(evaluate_value, f)
    num_list = []  # 迭代代数
    for episode_num, _ in enumerate(evaluate_value):
        num_list.append(episode_num)

    data_dict = {f'{file_name}_num_list':num_list,f'{file_name}_evaluate_value':evaluate_value}
    savemat(f'./result/data/{file_name}_data.mat',data_dict)

    plt.plot(num_list, evaluate_value)
    plt.title('evaluate of agents')
    plt.xlabel('episodes')
    plt.ylabel('evaluate')
    plt.legend()  # 图例可以进一步设置
    plt.savefig('./result/figure/{}_evaluate_of_agents_{}.png'.format(file_name,formatted_time))

    plt.show()





# 缓存区测试
if __name__ == '__main__':
    replay_buffer = ReplayBuffer(5)
    for i in range(5):
        replay_buffer.add(i,i+1,i+2,i+3,i+4,i+5)
    print(replay_buffer.sample(5))
    print(tuple_t([(1,3),(2,3),(4,5)]))
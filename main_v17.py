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
import setting
from algorithms import DDPG, MADDPG
import pickle
from MARL_env import PacketRoutingEnv
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

def single_step_test():
    agents_node = [1, 2, 4, 5, 6]
    agents_num = len(agents_node)
    state_dim = 4
    critic_input_dim = (state_dim + 1) * agents_num
    critic_hidden_dim = agents_num
    actor_lr = 0.0001
    critic_lr = 0.001
    device = "cuda" if torch.cuda.is_available() else "cpu"
    node_id = 1
    agents = DDPG(node_id, state_dim, critic_input_dim, critic_hidden_dim, actor_lr, critic_lr, device)
    with open('sample.pkl', 'rb') as f:
        sample = pickle.load(f)
    states, actions, rewards, next_states, done = sample

    gamma = 0.95
    tau = 1e-2
    env = PacketRoutingEnv(agents_node)
    maddpg = MADDPG(env, device, actor_lr, critic_lr, state_dim, critic_input_dim, critic_hidden_dim, gamma, tau)
    for a_i in range(agents_num):
        maddpg.update(sample, a_i)
    maddpg.update_all_targets()

    # TODO: 写一个函数，丢进去一个状态，返回邻接矩阵和属性矩阵


# TODO: 20240406 星期六
# TODO：完成训练过程
# TODO：在main.py中写一个函数evaluate，用于对学习的策略进行评估，此时不会进行探索
def evaluate(env, maddpg, n_episode=10, episode_length=25):
    returns = np.zeros(len(env.agents_node))
    for _ in range(n_episode):
        env.reset()
        for t_i in range(episode_length):
            agents = maddpg.agents
            _, acts, rews, _, done = env.step(agents, explore=False)
            rews = np.array(rews)
            returns += rews / n_episode
            if done:
                print('到达终点')
                break
        else:
            returns += setting.terminal_reward/setting.MAX_TASK*(setting.MAX_TASK-
                        sum(env.node_package_num_dict.values()))/n_episode
    return returns.tolist()


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


    logger.info(f'\n全局参数\n流量矩阵:\n{setting.traffic_matrix}'
          f'\n奖励类型:\n{setting.reward_type}'
          f'\n探索率:\n{setting.eps}')
    ########

    agents_node = [1, 2, 4, 5, 6]
    agents_num = len(agents_node)
    state_dim = 4
    critic_input_dim = (state_dim + 1 - 2) * agents_num + 2  # 在评价网络输入中去掉重复的量
    critic_hidden_dim = agents_num


    ### 超参数设置
    actor_lr = 0.002           # 动作网络学习率
    critic_lr = 0.00002        # 评价网络学习率
    gamma = 0.95               # 折扣因子
    tau = 1e-2                 # 软学习因子
    buffer_size = 100000       # 缓冲区大小用于保存数据,缓冲区存在数据删除，会导致运行速度变慢
    minimal_size = 4000        # 最小缓冲区大小，用于控制
    batch_size = 8             # 批量大小
    update_interval = 10       # 更新步数
    num_episodes = 300         # 序列数
    episode_length = 500       # 每条序列的最大长度
    eva_length = 500           # 测试序列长度
    eva_episode = 1            # 评价周期数
    min_save_episode = 100     # 最小保存网络代数
    save_episode = 5           # 保存间隔

    setting.adjust_learning_rate_limit = actor_lr*0.001

    logger.info(f'\n超参数\n'
          f'\n动作网络学习率:\n{actor_lr}'
          f'\n评价网络学习率:\n{critic_lr}'
          f'\n折扣因子:\n{gamma}'
          f'\n缓冲区大小:\n{buffer_size}'
          f'\n最小缓冲区大小:\n{minimal_size}'
          f'\n批量大小:\n{batch_size}'
          f'\n更新步数:\n{update_interval}'
          f'\n序列数:\n{num_episodes}'
          f'\n每条序列的最大长度:\n{episode_length}'
          f'\n测试序列长度:\n{eva_length}'
          f'\n评价周期数:\n{eva_episode}'
          f'\n')
    #######################################

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # node_id = 1
    # alpha = 0.9
    replay_buffer = ReplayBuffer(buffer_size)  # 建立缓冲区
    env = PacketRoutingEnv(agents_node)
    maddpg = MADDPG(env, device, actor_lr, critic_lr, state_dim, critic_input_dim, critic_hidden_dim, gamma, tau)
    # n_episodes = 1  # 评价信息显示间隔
    eva = Evaluate(eva_episode,eva_length)
    return_list = []  # 记录每一轮的回报(return)
    total_step = 0

    for i_episode in range(num_episodes):
        env.reset()
        agents = maddpg.agents
        done = False
        for e_i in range(episode_length):
            sample = env.step(agents)
            obs, acts, rews, next_obs, done = sample
            if done:
                replay_buffer.add(obs, acts, rews, next_obs, done)
                break
            elif e_i == episode_length:   # 提前到达终点
                # rews += setting.terminal_reward/setting.MAX_TASK*(setting.MAX_TASK-
                #         sum(env.node_package_num_dict.values()))   # 目标导向
                rews += setting.unit_reward*(setting.MAX_TASK-sum(env.node_package_num_dict.values()))
                replay_buffer.add(obs, acts, rews, next_obs, done)
            else:
                replay_buffer.add(obs, acts, rews, next_obs, done)

            total_step += 1
            if replay_buffer.size() >= minimal_size and total_step % update_interval == 0:
                sample_batch = replay_buffer.sample(batch_size)
                    # 将数据转化batchsize的更新形式
                    # 获取到的样本有自己的数据类型，需要对其做处理。
                    # 需要对模型进行测试
                    #         with open('samples.pkl','wb') as f:
                    #             pickle.dump(sample_batch,f)
                    # if i_episode == 6:  # 测试所用
                    #     break
                for a_i in range(len(agents)):
                    maddpg.update(sample_batch, a_i)
                maddpg.update_all_targets()

        if i_episode >= min_save_episode and i_episode % save_episode == 0:
            for a_i in range(len(agents)):
                torch.save(maddpg.agents[a_i].actor.state_dict(), f'result/model/{file_name}_actor{i_episode}-{a_i}.pth')
                torch.save(maddpg.agents[a_i].critic.state_dict(), f'result/model/{file_name}_critic{i_episode}-{a_i}.pth')
                torch.save(maddpg.agents[a_i].target_actor.state_dict(),
                           f'result/model/{file_name}_target_actor{i_episode}-{a_i}.pth')
                torch.save(maddpg.agents[a_i].target_critic.state_dict(),
                           f'result/model/{file_name}_target_critic{i_episode}-{a_i}.pth')

                if i_episode == num_episodes:
                    for actor_param_group, critic_param_group in zip(maddpg.agents[a_i].actor_optimizer.param_groups,
                                                                     maddpg.agents[a_i].critic_optimizer.param_groups):
                        logger.info("最终动作网络学习率", actor_param_group['lr'])
                        logger.info("最终评价网络学习率", critic_param_group['lr'])

        # if (i_episode + 1) % n_episodes == 0:
        #     actor_lr *= alpha
        #     critic_lr *= alpha
        #     ep_returns = evaluate(env, maddpg, n_episodes, test_length)
        #     return_list.append(ep_returns)
        #     logger.info(f"Episode: {i_episode + 1},{ep_returns}")

        ep_returns = eva.evaluate(env,maddpg,i_episode)
        return_list.append(ep_returns)
        logger.info(f"Episode: {i_episode + 1},{ep_returns}")

    convergence_result_display(return_list)

    # TODO： 不收敛，一个原因没有目标，添加终端奖励5000。
    # TODO： 修改一下每代训练结束的条件，增加每代训练的次数。

    # TODO： 20240407 星期日
    # TODO： 评价网络是更新的
    # TODO： 发现问题，在动作网络更新时没有进行部分观测
    # TODO： 在algorithms.py中写一个函数partial_observation实现部分观测.
    # TODO： 将gumbel采样的函数从algorithms.py中DDPG类中独立出来，放到comments下的utils.py中，供其他函数调用
    # TODO:  设计一个奖励函数，使得算法收敛。
    # TODO:  在MARL_env.py中写一个函数reward_calculate()来计算奖励函数。
    # TODO:  将缓冲区空的情况的收益记为零，对于能量比较高的情况下仅仅记录第一次。
    # TODO： 排除发送空数据包的情况。
    # TODO:  不稳定，可能是更新尺度太大。

    # TODO:  20240409
    # TODO:  在文件夹comments中utils.py中添加一个log_management函数，用于管理运行日志。
    # TODO:  在main.py中写一个函数convergence_result_display, 用于绘制收敛图。

    # TODO：  20240410
    # TODO：  使用批量训练的方法(先备份一下现在的代码)

    # TODO:  20140414
    # TODO:  修改data_trans.py下的类GetState下的to_Data函数，将x的命名改为features,去掉node_ids这个分量。
    # TODO： 规范一下数据格式

    # TODO：20240418 星期四
    # TODO：经过测试，程序收敛时跳跃过大，在做以下调整
    #  之前的可变学习率未生效，进行调整，在到达终点后每10步降低5%，直到到达最低学习率。
    # TODO：在main.py中新增学习率调整参数，将函数evaluate扩展为可以迭代的类，写到utils中。
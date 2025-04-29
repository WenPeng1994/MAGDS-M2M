#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Auther:Evans_WP
@Contact:WenPeng19943251019@163.com
@Time:2024/6/30 17:07
@File:training_continue.py
@Desc:****************
"""
# 继续训练函数

# 需要的数据-模型(类-值)，数据(经验池)
import copy
import torch
import setting
from algorithms import  MADDPG
from MARL_env import PacketRoutingEnv
import numpy as np
from comments.utils import log_management,ReplayBuffer,Evaluate,convergence_result_display
import time
import random
import os
import pickle
# 全局设置随机种子
seed = 1234
# seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    # 新增需要修改的参数
    file_name = 'main_v128'
    add_train_epsilon = 500  # 增加训练的周期数
    model_ver = copy.deepcopy(file_name)


    # 下面拷贝file_name中的数据
    #############################################################
    formatted_time = time.strftime("%Y-%m-%d_%H:%M", time.localtime())
    logger = log_management(file_name)
    logger.info(f'\n\n{formatted_time}{file_name}运行结果\n\n')

    ### main_v11.py后将所有参数全部放在main中
    setting.traffic_matrix = setting.traffic_matrix_4  # 流量矩阵
    setting.reward_type = 12  # 奖励类型
    setting.eps = 0.1  # 探索率
    setting.beta = 0.99  # 时延占比
    setting.terminal_reward = 0  # 终端奖励
    setting.unit_reward = setting.terminal_reward ** (1 / setting.MAX_TASK)  # 这两个值是绑定的
    setting.xml_path = './topo.xml'

    logger.info(f'\n全局参数\n流量矩阵:\n{setting.traffic_matrix}'
                f'\n奖励类型:\n{setting.reward_type}'
                f'\n探索率:\n{setting.eps}'
                f'\n时延占比\n{setting.beta}'
                f'\n终端奖励\n{setting.terminal_reward}')
    ########

    agents_node = [1, 2, 4, 5, 6]  # 修改重要性节点
    agents_num = len(agents_node)
    state_dim = 4
    critic_input_dim = (state_dim + 1 - 2) * agents_num + 2  # 在评价网络输入中去掉重复的量
    critic_hidden_dim = agents_num

    ### 超参数设置
    actor_lr = 0.002  # 动作网络学习率
    critic_lr = 0.00002  # 评价网络学习率
    gamma = 0.95  # 折扣因子
    tau = 1e-2  # 软学习因子
    buffer_size = 10000000  # 缓冲区大小用于保存数据,缓冲区存在数据删除，会导致运行速度变慢
    minimal_size = 4000  # 最小缓冲区大小，用于控制
    # minimal_size = 40  # 最小缓冲区大小，用于控制
    batch_size = 8  # 批量大小
    update_interval = 10  # 更新步数
    num_episodes = 1500  # 序列数
    episode_length = 500  # 每条序列的最大长度
    eva_length = 500  # 测试序列长度
    eva_episode = 1  # 评价周期数
    min_save_episode = 0  # 最小保存网络代数
    save_episode = 5  # 保存间隔

    setting.adjust_learning_rate_limit = actor_lr * 0.001
    setting.adjust_learning_rate_range = 1 - 0.02

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
    #############################################################
    # 新增逻辑，自动识别内存中最新的周期数，读取数据buffer和model
    buffer_path = './result/buffer'
    max_epsion_num = 0
    for filename in os.listdir(buffer_path):
        # 判断文件名是file_name开头
        if filename.startswith(file_name):
            current_epsion_num = int(filename.split('-')[1])
            max_epsion_num = current_epsion_num if current_epsion_num > max_epsion_num else max_epsion_num
    traning_continue_buffer_name = f'./result/buffer/{file_name}-{max_epsion_num}-replay_buffer.pkl'
    assert os.path.exists(traning_continue_buffer_name), f'文件中不存在这个文件{file_name}-{max_epsion_num}-replay_buffer.pkl，请检查逻辑'
    new_num_episodes = max_epsion_num + add_train_epsilon

    # 读取学习率
    with open(f'./result/learning_rate/{file_name}_lr.pkl', 'rb') as f:
        final_lr = pickle.load(f)

    actor_lr = final_lr['actor_lr']  # 动作网络学习率
    critic_lr = final_lr['critic_lr']  # 评价网络学习率


    #############################################################
    replay_buffer = ReplayBuffer(buffer_size)  # 建立缓冲区
    replay_buffer.load(traning_continue_buffer_name)

    env = PacketRoutingEnv(agents_node)
    maddpg = MADDPG(env, device, actor_lr, critic_lr, state_dim, critic_input_dim, critic_hidden_dim, gamma, tau)
    # n_episodes = 1  # 评价信息显示间隔
    eva = Evaluate(eva_episode, eva_length)

    return_list = []  # 记录每一轮的回报(return)

    # 模型读取
    model_eps = max_epsion_num-1
    for i,agt in enumerate(maddpg.agents):
        # 构造模型路径
        model_path = './result/model/' + model_ver + '_actor' + str(model_eps) + '-' + str(i) + '.pth'
        model_parameters = torch.load(model_path)
        agt.actor.load_state_dict(model_parameters)


    for i_episode in range(max_epsion_num,new_num_episodes,1):
        env.reset()
        total_step = 0
        agents = maddpg.agents
        done = False
        for e_i in range(episode_length):
            sample = env.avoide_loop_step(agents)
            obs, acts, rews, next_obs, done, precursors = sample
            if done:
                replay_buffer.add(obs, acts, rews, next_obs, done, precursors)
                break
            # elif e_i == episode_length:   # 提前到达终点
            #     # rews += setting.terminal_reward/setting.MAX_TASK*(setting.MAX_TASK-
            #     #         sum(env.node_package_num_dict.values()))   # 目标导向
            #     rews += setting.unit_reward**(setting.MAX_TASK-sum(env.node_package_num_dict.values()))
            #     replay_buffer.add(obs, acts, rews, next_obs, done, precursors)
            else:
                replay_buffer.add(obs, acts, rews, next_obs, done, precursors)

            # if replay_buffer.size() >= minimal_size:
            #     print(replay_buffer.size())

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

        if (i_episode >= min_save_episode and i_episode % save_episode == 0) or i_episode > num_episodes-10:
            for a_i in range(len(agents)):
                torch.save(maddpg.agents[a_i].actor.state_dict(), f'result/model/{file_name}_actor{i_episode}-{a_i}.pth')
                torch.save(maddpg.agents[a_i].critic.state_dict(), f'result/model/{file_name}_critic{i_episode}-{a_i}.pth')
                torch.save(maddpg.agents[a_i].target_actor.state_dict(),
                           f'result/model/{file_name}_target_actor{i_episode}-{a_i}.pth')
                torch.save(maddpg.agents[a_i].target_critic.state_dict(),
                           f'result/model/{file_name}_target_critic{i_episode}-{a_i}.pth')

        if i_episode == num_episodes-1:
            for actor_param_group, critic_param_group in zip(maddpg.agents[0].actor_optimizer.param_groups,
                                                             maddpg.agents[0].critic_optimizer.param_groups):
                # 这里每个模型的学习率是一样的，所以直接取第一个
                logger.info(f"最终动作网络学习率:{actor_param_group['lr']}")
                logger.info(f"最终评价网络学习率:{critic_param_group['lr']}")
                # 保存最终学习率
                final_lr = {'actor_lr': actor_param_group['lr'], 'critic_lr': critic_param_group['lr']}
                with open(f'./result/learning_rate/{file_name}_lr.pkl', 'wb') as f:
                    pickle.dump(final_lr, f)

                            # if (i_episode + 1) % n_episodes == 0:
        #     actor_lr *= alpha
        #     critic_lr *= alpha
        #     ep_returns = evaluate(env, maddpg, n_episodes, test_length)
        #     return_list.append(ep_returns)
        #     logger.info(f"Episode: {i_episode + 1},{ep_returns}")

        ep_returns = eva.evaluate(env,maddpg,i_episode)
        return_list.append(ep_returns)
        logger.info(f"Episode: {i_episode + 1},{ep_returns},residual_packets:{sum(env.node_package_num_dict.values())},"
                    f"num_step:{total_step}")


    convergence_result_display(return_list,file_name)
    replay_buffer.save(f'result/buffer/{file_name}-{new_num_episodes}-replay_buffer.pkl')


























    for model_eps_num in range(max_epsion_num-1, new_num_episodes, save_episode):
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
            model_name = f'main_{model_ver}_{model_eps}_episode'
            time_statistic = node_package_display(node_package_nums_dict, model_name, display=False)
            var_energy = node_energy_display(node_energy_use_dict, model_name, display=False)
            time_statistic_list.append(time_statistic)
            var_energy_list.append(var_energy)
            episodes_list.append(model_eps_num)
        else:
            continue
    model_evaluate_display(episodes_list, time_statistic_list, var_energy_list, model_ver)
else:
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
        if alg_name == "magddpg":
            node_package_num_dict, node_energy_use_dict, link_msg_dict, done = env.MARL_test_step(maddpg.agents,
                                                                                                  explore=False)
        else:
            node_package_num_dict, node_energy_use_dict, link_msg_dict, done = env.other_alg_step(algorithm)
        for node in env.G.nodes:
            node_package_nums_dict[node].append(node_package_num_dict[node])
        step += 1

    # 3.绘制图
    model_name = f'main_{model_ver}_{model_eps}_episode'
    node_package_display(node_package_nums_dict, model_name)
    node_energy_display(node_energy_use_dict, model_name)
    link_msg_display(link_msg_dict, model_name)





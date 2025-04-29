#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Auther:Evans_WP
@Contact:WenPeng19943251019@163.com
@Time:2024/3/5 11:02
@File:algorithms.py
@Desc:算法仿真
"""
import copy
import pickle

import networkx as nx
import random

import self as self
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from comments.utils import gumbel_softmax,gumbel_softmax_sample,onehot_from_logits,sample_gambel,\
    gumbel_softmax_batch,gumbel_softmax_sample_batch,onehot_from_logits_batch,sample_gambel_batch,\
    onehot_from_logits_max, random_one_in_list, tuple_t
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader


import setting
from data_trans import GetState
import numpy as np

# 1.随机寻找下一跳
def random_find_next_hop(G, current_hop, std, importance_statistic=False):
    """
    运用随机的方法寻找下一跳的位置
    :param current_hop: 当前的位置
    :param std: 目的地
    :return:
    """
    # 1.根据当前的位置和目的地获取图的所有简单路径
    if std:
        all_paths = list(nx.all_simple_paths(G, source=current_hop, target=std))
        next_hop_list = [path[1] for path in all_paths]
        next_hop = random.choice(next_hop_list)
        if importance_statistic:   # 需要统计重要性
            return next_hop,len(set(next_hop_list))-1
        # if importance_statistic:
        #     return next_hop,len(list(G.neighbors(current_hop))) - 2

    else:
        neighbors = G.neighbors(current_hop)
        list_neighboers = list(neighbors)
        next_hop = random.choice(list_neighboers)
        if importance_statistic:
            return next_hop, 0
    return next_hop

# 2.SPR,最短路径寻找下一跳
def random_short_path_hop(G, current_hop, std,importance_statistic=False,randomness=False):
    """
    随机最短路径寻找下一跳，注意这里可能出现多条最短路径，所以需要进行随机
    # 注意在之前的算法中不要给importance_statistic赋值，否者可能会报错
    :param G:
    :param current_hop:
    :param std:
    :return:
    """
    if randomness:  # 固定随机性
        random.seed(1234)
        if std:
            all_paths = list(nx.all_shortest_paths(G,source=current_hop,target=std))
            next_hop_list = [path[1] for path in all_paths]
            next_hop = random.choice(next_hop_list)
            if importance_statistic:
                return next_hop, len(set(next_hop_list))-1
            # if importance_statistic:
            #     return next_hop, len(list(G.neighbors(current_hop))) - 2
        else:
            neighbors = G.neighbors(current_hop)
            list_neighboers = list(neighbors)
            next_hop = random.choice(list_neighboers)
            if importance_statistic:
                return next_hop, 0
        return next_hop
    else:
        if std:
            all_paths = list(nx.all_shortest_paths(G, source=current_hop, target=std))
            next_hop_list = [path[1] for path in all_paths]
            next_hop = random.choice(next_hop_list)
            if importance_statistic:
                return next_hop, len(set(next_hop_list)) - 1
            # if importance_statistic:
            #     return next_hop, len(list(G.neighbors(current_hop))) - 2
        else:
            neighbors = G.neighbors(current_hop)
            list_neighboers = list(neighbors)
            next_hop = random.choice(list_neighboers)
            if importance_statistic:
                return next_hop, 0
        return next_hop

# 避免回路的最短路径算法
def avoid_loop_short_path(G, current_hop, std, precursor, importance_statistic=False):
    # assert precursor in list(G.neighbors(current_hop)) or precursor == None, f'前驱节点{precursor}不在邻居节点{list(G.neighbors(current_hop))}中，请检查bug！'
    # if current_hop==2:
    #      print(f'节点{current_hop}的前驱{precursor},目标节点{std}')
    if std:
        all_paths = list(nx.all_shortest_paths(G, source=current_hop, target=std))  # 获取所有最短路径
        next_hop_list = [path[1] for path in all_paths]  # 获取所有最短路径的下一跳
        next_hop = random.choice(next_hop_list)  # 随机选择下一跳
        while next_hop == precursor: # 下一跳就是前驱节点的话，进行循环
            next_hop = random.choice(next_hop_list)
            if len(set(next_hop_list)) == 1: # 迫不得已的情况下选择回路
                break
        if importance_statistic:   # 随机性的统计，只要随机性发生就加一，随机等价性
            # 什么是随机性，在去回路的基础上，还有不同选择的情况下，称为有随机性选择
            random_selection_set = set(next_hop_list) - set([current_hop])
            randomness = 1 if len(random_selection_set) > 1 else 0  # 如果除了回路还存在多个选择，那么就算一个随机性决策。
            return next_hop, randomness
        # if importance_statistic:
        #     return next_hop, len(list(G.neighbors(current_hop))) - 2
    else:
        neighbors = G.neighbors(current_hop)
        list_neighboers = list(neighbors)
        next_hop = random.choice(list_neighboers)
        if importance_statistic:
            return next_hop, 0
    return next_hop


def avoid_loop_random_path(G, current_hop, std, precursor, importance_statistic=False):
    # 1.根据当前的位置和目的地获取图的所有简单路径
    if std:
        all_paths = list(nx.all_simple_paths(G, source=current_hop, target=std))
        next_hop_list = [path[1] for path in all_paths]
        next_hop = random.choice(next_hop_list)
        while next_hop == precursor: # 下一跳就是前驱节点的话，进行循环
            next_hop = random.choice(next_hop_list)
            if len(set(next_hop_list)) == 1: # 迫不得已的情况下选择回路
                break
        if importance_statistic:  # 随机性的统计，只要随机性发生就加一，随机等价性
            # 什么是随机性，在去回路的基础上，还有不同选择的情况下，称为有随机性选择
            random_selection_set = set(next_hop_list) - set([current_hop])
            randomness = 1 if len(random_selection_set) > 1 else 0  # 如果除了回路还存在多个选择，那么就算一个随机性决策。
            return next_hop, randomness
        # if importance_statistic:
        #     return next_hop,len(list(G.neighbors(current_hop))) - 2
    else:
        neighbors = G.neighbors(current_hop)
        list_neighboers = list(neighbors)
        next_hop = random.choice(list_neighboers)
        if importance_statistic:
            return next_hop, 0
    return next_hop



# 3. GCN-MAAC算法
# 1) GcnNet
#  GCN层的定义
class GraphConvolution(nn.Module):
    def __init__(self,input_dim,output_dim,use_bias=True):
        """图卷积：L*X*\theta
        Args:
        ----------
            input_dim: int
                节点输入特征的维度
            output_dim: int
                输出特征维度
            use_bias: bool,optional
                是否使用偏置
        """
        super(GraphConvolution,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim,output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias',None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        模型参数初始化
        :return:
        """
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self,adjacency,input_feature):
        """
        邻接矩阵是稀疏矩阵，因此在计算时使用稀疏矩阵乘法
        :param adjacency: 邻接矩阵
        :param input_feature: 输入特征
        :return:
        """

        support = torch.matmul(input_feature,self.weight)
        output = torch.matmul(adjacency,support)
        if self.use_bias:
            output += self.bias
        return output

# 两层GCN模型,用于构建动作网络
class GCNActor(nn.Module):
    """
    定义一个包含两层GraphConvolution的模型，用于构建动作网络
    """
    def __init__(self,input_dim=2):
        super(GCNActor,self).__init__()
        self.gcn1 = GraphConvolution(input_dim,input_dim)
        self.gcn2 = GraphConvolution(input_dim,1)

    def forward(self,adjacency,feature):
        h = F.relu(self.gcn1(adjacency,feature))
        logits = self.gcn2(adjacency,h)
        return logits

# 两层GCN模型,用于构建动作网络
class GCNActorBatch(nn.Module):
    """
    定义一个包含两层GraphConvolution的模型，用于构建动作网络
    """
    def __init__(self,input_dim=2):
        super(GCNActorBatch,self).__init__()
        self.gcn1 = GraphConvolution(input_dim,input_dim)
        self.gcn2 = GraphConvolution(input_dim,1)

    def forward(self,data,device):
        adjacency, features = data.adjacency.to(device), data.features.to(device)
        # if adjacency.size()
        h = F.relu(self.gcn1(adjacency,features))
        logits = self.gcn2(adjacency,h)
        return logits




# GNN模型，用于构建评价网络
class GNNCritic(nn.Module):
    """
    定义一个包含两层GraphConvolution的模型，用于构建评价网络
    """
    def __init__(self,input_dim=2,hidden_dim=16):
        super(GNNCritic, self).__init__()
        self.conv1 = GraphConvolution(input_dim, input_dim)
        self.lin1 = nn.Linear(input_dim,hidden_dim)
        self.lin2 = nn.Linear(hidden_dim,hidden_dim)
        self.lin3 = nn.Linear(hidden_dim,1)

    def forward(self,adjacency,feature):
        out = F.relu(self.conv1(adjacency, feature))
        x = out + feature
        x = torch.sum(x, dim=0)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x

# GNN模型，用于构建评价网络
class GNNCriticBatch(nn.Module):
    """
    定义一个包含两层GraphConvolution的模型，用于构建评价网络
    """
    def __init__(self,input_dim=2,hidden_dim=16):
        super(GNNCriticBatch, self).__init__()
        self.conv1 = GraphConvolution(input_dim, input_dim)
        self.lin1 = nn.Linear(input_dim,hidden_dim)
        self.lin2 = nn.Linear(hidden_dim,hidden_dim)
        self.lin3 = nn.Linear(hidden_dim,1)

    def forward(self,data,device):
        adjacency, features = data.adjacency.to(device), data.features.to(device)
        out = F.relu(self.conv1(adjacency, features))
        # print(out)
        x = out + features
        x = torch.sum(x, dim=1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x


# 定义DDPG
class DDPG:
    """
    基于图神经网络的DDPG算法
    """
    def __init__(self,node_id,state_dim,critic_input_dim,critic_hidden_dim,actor_lr,critic_lr,device):
        # 节点iD
        self.node_id = node_id
        # 策略网络
        self.actor = GCNActorBatch(state_dim).to(device)
        # 目标策略网络
        self.target_actor = GCNActorBatch(state_dim).to(device)
        # 评价网络
        self.critic = GNNCriticBatch(critic_input_dim,critic_hidden_dim).to(device)
        # 目标评价网络
        self.target_critic = GNNCriticBatch(critic_input_dim,critic_hidden_dim).to(device)
        # 用原网络的参数初始化目标网络的参数
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 定义迭代器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=critic_lr)
        self.device = device

    def take_action(self,state,explore=False):  # state 是一个图类
        state_data = state.to_Data()
        # adjacency = state_data.adjacency.to(self.device)
        # feature = state_data.x.to(self.device)
        model = copy.deepcopy(self.actor)   # 不做训练的时候使用
        model.eval()
        # logits = model(adjacency, feature.T)   # 将转化后的数据放入GCN,获得的是一个Tensor张量
        logits = model(state_data,self.device)
        # # 对动作进行局部观测，找到源节点的位置
        # _, neighbor_hot = state.get_neighbors()
        #
        # logits_local = logits[neighbor_hot].T
        #
        # # 通过epsilon-贪心算法获取动作
        # if explore:
        #     action = gumbel_softmax(logits_local)   # 可以进行梯度回传
        # else:
        #     action = onehot_from_logits(logits_local)  # 不可以进行梯度回传
        #
        # action_board = torch.zeros([len(neighbor_hot), 1], dtype=logits_local.dtype, device=self.device)
        # action_board[neighbor_hot,0] = action.view(-1)
        action_board = partial_observation(logits,state,explore)
        action = action_board.detach().cpu().view(-1).numpy()
        if sum(action) != 1:
            print('动作{}不是独热编码'.format(action))
            # action = random_one_in_list(action)

        assert sum(action) == 1, '动作{}不是独热编码'.format(action)
        return action_board


    def avoid_loop_take_action(self,state,precursor,explore=False):  # state 是一个图类
        state_data = state.to_Data()
        # adjacency = state_data.adjacency.to(self.device)
        # feature = state_data.x.to(self.device)
        model = copy.deepcopy(self.actor)   # 不做训练的时候使用
        model.eval()
        # logits = model(adjacency, feature.T)   # 将转化后的数据放入GCN,获得的是一个Tensor张量
        logits = model(state_data,self.device)
        # # 对动作进行局部观测，找到源节点的位置
        # _, neighbor_hot = state.get_neighbors()
        #
        # logits_local = logits[neighbor_hot].T
        #
        # # 通过epsilon-贪心算法获取动作
        # if explore:
        #     action = gumbel_softmax(logits_local)   # 可以进行梯度回传
        # else:
        #     action = onehot_from_logits(logits_local)  # 不可以进行梯度回传
        #
        # action_board = torch.zeros([len(neighbor_hot), 1], dtype=logits_local.dtype, device=self.device)
        # action_board[neighbor_hot,0] = action.view(-1)
        action_board = avoid_loop_partial_ob(logits,state,precursor,explore)
        action = action_board.detach().cpu().view(-1).numpy()
        if sum(action) != 1:
            print('动作{}不是独热编码'.format(action))
            # action = random_one_in_list(action)

        assert sum(action) == 1, '动作{}不是独热编码'.format(action)
        return action_board


    def actor_evaluation(self,state_act_data):
        """
        动作评价函数
        :param state_act_data: 由状态动作Data类型的数据，其中包含属性和邻接矩阵
        :return: 评价值
        """
        adjacency = state_act_data.adjacency.to(self.device)
        feature = state_act_data.x.to(self.device)
        model = copy.deepcopy(self.critic)  # 不做训练的时候使用
        model.eval()
        critic_value = model(adjacency,feature.T)
        return critic_value


    def soft_update(self,net,target_net,tau):
        for param_taret,param in zip(target_net.parameters(),net.parameters()):
            param_taret.data.copy_(param_taret.data * (1.0-tau) + param.data * tau)


class MADDPG:
    def __init__(self,env,device,actor_lr,critic_lr,state_dim,critic_input_dim,critic_hidden_dim,gamma,tau):
        self.agents = []
        for i,node_id in enumerate(env.agents_node):  # 需要在环境中设计agents属性(包含重要节点的向量)
            self.agents.append(DDPG(node_id,state_dim,critic_input_dim,critic_hidden_dim,
                                    actor_lr,critic_lr,device))
        self.gamma = gamma
        self.tau = tau
        self.critic_criterion = torch.nn.MSELoss()
        self.device = device

    @property
    def policies(self):   # 多智能体策略
        return [agt.actor for agt in self.agents]

    @property
    def target_policies(self):  # 多智能目标策略
        return [agt for agt in self.agents]

    # def take_action(self,states,explore):
    #     """
    #     通过多智能体观测获取动作
    #     :param states: 这里的状态是观察到的图向量
    #     :param explore:
    #     :return:
    #     """
    #     # 1. 将图向量转化为张量向量
    #     states = [state.to_Data() for state in states]
    #     # 2. 通过

    # TODO: 20240403 星期三
    # TODO：修改algorithms.py 中类MADDPG中的update方法
    # 更新过程
    def single_update(self, sample, i_agent):
        obs, acts, rews, next_obs, done = sample
        cur_agent = self.agents[i_agent]
        # print(f'模型actor{i_agent}权重', cur_agent.target_actor.state_dict())

        cur_agent.critic_optimizer.zero_grad()

        # 1. 获取目标动作
        all_target_acts = [partial_observation(agt.target_actor(generate_actor_input_batch(_next_obs,self.device),self.device),_next_obs,explore=False) for
                           agt, _next_obs in zip(self.agents, next_obs)]

        # 2. 使用评价网络和目标评价网络获得值和目标值
        target_critic_input_data = generate_critic_input(next_obs,all_target_acts,self.device)
        target_critic_value = rews[i_agent] + self.gamma * cur_agent.target_critic(*target_critic_input_data) * (1-done)
        critic_input = generate_critic_input(obs,acts,self.device)
        critic_value = cur_agent.critic(*critic_input)
        # print(f'模型critic{i_agent}权重', cur_agent.critic.state_dict())

        # 3. 通过奖励值更新评价网络
        critic_loss = self.critic_criterion(critic_value,target_critic_value.detach())
        critic_loss.backward()
        cur_agent.critic_optimizer.step()

        # 4. 通过值更新行动者网络
        cur_agent.actor_optimizer.zero_grad()
        actor_in = generate_actor_input(obs[i_agent],self.device)
        cur_actor_out = cur_agent.actor(*actor_in)
        # cur_act_vf_in = gumbel_softmax(cur_actor_out)
        all_actor_acs = []

        for i,(agent,ob) in enumerate(zip(self.agents,obs)):
            if i == i_agent:
                cur_act_vf_in = partial_observation(cur_actor_out,ob,explore=True)
                all_actor_acs.append(cur_act_vf_in)
            else:
                other_actor_vf_in = partial_observation(agent.actor(*generate_actor_input(ob,self.device)),ob,explore=False)
                all_actor_acs.append(other_actor_vf_in)

        vf_in = generate_critic_input(obs,all_actor_acs,self.device)
        actor_loss = -cur_agent.critic(*vf_in).mean()
        actor_loss += (cur_actor_out**2).mean()*1e-3
        actor_loss.backward()
        # # 检查梯度是否回传
        # for param in cur_agent.actor.parameters():
        #     if param.grad is not None:
        #         print("Gradient is backpropagated for parameter:", param)
        #     else:
        #         print("Gradient is not backpropagated for parameter:", param)
        cur_agent.actor_optimizer.step()

    # 更新过程
    def update(self, sample, i_agent):
        obs_batch, acts_batch, rews_batch, next_obs_batch, done_batch, precursors = sample
        cur_agent = self.agents[i_agent]
        # print(f'模型actor{i_agent}权重', cur_agent.target_actor.state_dict())

        cur_agent.critic_optimizer.zero_grad()

        # 1. 获取目标动作
        all_target_acts = []    # 5×2×15×1
        for agt,_next_obs_batch in zip(self.agents,next_obs_batch.T):  # 这里的转置
            actor_loader = generate_actor_input_batch(_next_obs_batch, self.device)
            for batch in actor_loader:
                all_target_acts.append(agt.actor(batch,self.device))


        # 2. 使用评价网络和目标评价网络获得值和目标值
        all_target_acts_tensor = torch.stack(all_target_acts,dim=0)
        all_target_acts_tensor = torch.transpose(all_target_acts_tensor,0,1)
        # 目标函数的输入
        target_critic_input_loader = generate_critic_input_batch(next_obs_batch, all_target_acts_tensor, self.device)
        target_critic_value = torch.unsqueeze(torch.tensor(rews_batch,device=self.device).T[i_agent],dim=0).T + self.gamma * torch.mul(cur_agent.target_critic(next(iter(target_critic_input_loader)),self.device) , (
                    1 - torch.unsqueeze(torch.tensor(done_batch,device=self.device,dtype=torch.float),dim=0).T))
        critic_input_loader = generate_critic_input_batch(obs_batch, acts_batch, self.device)
        critic_value = cur_agent.critic(next(iter(critic_input_loader)),self.device)
        # print(f'模型critic{i_agent}权重', cur_agent.critic.state_dict())

        # 3. 通过奖励值更新评价网络
        critic_loss = self.critic_criterion(critic_value, target_critic_value.detach())
        critic_loss.backward(retain_graph=True)
        cur_agent.critic_optimizer.step()

        # 4. 通过值更新行动者网络
        cur_agent.actor_optimizer.zero_grad()
        actor_in_loader = generate_actor_input_batch(obs_batch.T[i_agent], self.device)
        cur_actor_out = cur_agent.actor(next(iter(actor_in_loader)),self.device)
        cur_actor_out_batch = cur_actor_out.permute(0,2,1)
        # cur_act_vf_in = gumbel_softmax_batch(gumbel_softmax_in)
        all_actor_acs = []

        for i, (agent, ob_batch,precursor_list) in enumerate(zip(self.agents, obs_batch.T,tuple_t(precursors))):
            if i == i_agent:
                cur_act_vf_in = avoid_loop_partial_ob_batch(cur_actor_out_batch, ob_batch, precursor_list, explore=True)
                all_actor_acs.append(cur_act_vf_in)
            else:
                actor_in_loader = generate_actor_input_batch(obs_batch=ob_batch,device=self.device)
                other_actor_vf_in = avoid_loop_partial_ob_batch(agent.actor(next(iter(actor_in_loader)),self.device).permute(0,2,1), ob_batch, precursor_list,
                                                        explore=False)
                all_actor_acs.append(other_actor_vf_in)

        all_actor_acs = torch.stack(all_actor_acs,dim=0)
        all_actor_acs = torch.transpose(all_actor_acs,0,1)
        vf_in_loader = generate_critic_input_batch(obs_batch, all_actor_acs, self.device)
        actor_loss = -cur_agent.critic(next(iter(vf_in_loader)),self.device).mean()
        actor_loss += (cur_actor_out ** 2).mean() * 1e-3
        actor_loss.backward(retain_graph=True)
        # # 检查梯度是否回传
        # for param in cur_agent.actor.parameters():
        #     if param.grad is not None:
        #         print("Gradient is backpropagated for parameter:", param)
        #     else:
        #         print("Gradient is not backpropagated for parameter:", param)
        cur_agent.actor_optimizer.step()
    #
    # # TODO：构建一个函数，将状态和动作组成评价网络的输入。

    # TODO：利用软更新更新目标网络
    def update_all_targets(self):
        for agent in self.agents:
            agent.soft_update(agent.actor,agent.target_actor,self.tau)
            agent.soft_update(agent.critic,agent.target_critic,self.tau)

# TODO: 20240405,构建一个函数，将状态和动作组成评价网络的输出
def generate_critic_input(obs,acts,device):
    """
    生成评价网络的输入
    :param obs:
    :param acts:
    :return:
    """
    for ob,act in zip(obs,acts):
        if not torch.is_tensor(act):
            act = torch.tensor(act)
        ind_agent = torch.cat((ob.to_Data().x.to(device),torch.unsqueeze(act.view(-1),dim=0).to(device)),dim=0)
        if 'x' not in locals():
            x = ind_agent
        else:
            x = torch.cat((x,ind_agent),dim=0)
        if 'adjacency' not in locals():
            adjacency = ob.to_Data().adjacency.to(device)
        if 'node_ids' not in locals():
            node_ids = ob.G.nodes
    return adjacency, x.T


def generate_actor_input(ob,device):
    """
    生成行动者网络的输入
    :param ob: 状态
    :return:
    """
    data = ob.to_Data()
    adjacency = data.adjacency.to(device)
    feature = data.x.T.to(device)
    return adjacency,feature

def partial_observation(act_logits,ob,explore=False):
    """
    对动作进行部分观测，并保持梯度可传递
    :param act_logits: 动作的逻辑值
    :param ob: 状态
    :return: act
    """
    _,neighbor_hot = ob.get_neighbors()
    act_local = act_logits[neighbor_hot].T
    # 通过epsilon-贪心算法获取动作
    if explore:
        action = gumbel_softmax(act_local)  # 可以进行梯度回传
    else:
        action = onehot_from_logits_max(act_local)  # 不可以进行梯度回传

    action_board = torch.zeros([len(neighbor_hot), 1], dtype=act_logits.dtype, device=act_logits.device)
    action_board[neighbor_hot, 0] = action.view(-1)
    return action_board


# 避免回路进行观测
def avoid_loop_partial_ob(act_logits,ob,precursor,explore=False):
    """
    在避免回路的情况下，对动作进行部分观测，并保持梯度可传递
    :param act_logits: 动作的逻辑值
    :param ob: 状态
    :precursor: 前驱
    :return: act
    """
    neighbors, neighbor_hot = ob.get_neighbors()
    for node_index,node in enumerate(ob.node_ids):
        if node == precursor and len(neighbors)>1:
            neighbor_hot[node_index] = 0

    # 动作修正
    for next_hop in neighbors:
        if len(list(ob.G.neighbors(next_hop))) == 1 and ob.node_id in list(
                ob.G.neighbors(next_hop)) and setting.action_modification:
            neighbor_hot[ob.node_ids == next_hop] = 0
            # 动作修正测试 -- 通过
            # print('动作修正')

    act_local = act_logits[neighbor_hot].T
    if act_local.size()[1] == 0:  # 无路可走，原路返回
        for node_index, node in enumerate(ob.node_ids):
            if node == precursor and len(neighbors) > 1:
                neighbor_hot[node_index] = 1
                act_local = act_logits[neighbor_hot].T

    #通过epsilon-贪心算法获取动作
    if explore:
        action = gumbel_softmax(act_local)  # 可以进行梯度回传
    else:
        action = onehot_from_logits_max(act_local)  # 不可以进行梯度回传

    action_board = torch.zeros([len(neighbor_hot), 1], dtype=act_logits.dtype, device=act_logits.device)
    action_board[neighbor_hot, 0] = action.view(-1)
    return action_board


def partial_observation_batch(act_logits,ob_batch,explore=False):
    """
    对动作进行部分观测，并保持梯度可传递
    :param act_logits: 动作的逻辑值
    :param ob: 状态
    :return: act
    """
    action_list = []
    for act_logit,ob in zip(act_logits,ob_batch):
        _,neighbor_hot = ob.get_neighbors()

        act_local = torch.unsqueeze(act_logit[0,neighbor_hot],dim=0)
        # 通过epsilon-贪心算法获取动作
        if explore:
            action = gumbel_softmax_batch(act_local)  # 可以进行梯度回传
        else:
            action = onehot_from_logits_batch(act_local)  # 不可以进行梯度回传

        action_board = torch.zeros([len(neighbor_hot), 1], dtype=act_logits.dtype, device=act_logits.device)
        action_board[neighbor_hot, 0] = action.view(-1)
        action_list.append(action_board)
    action_tensor = torch.stack(action_list,dim=0)
    return action_tensor


def avoid_loop_partial_ob_batch(act_logits,ob_batch,precursor_list,explore=False):
    """
    对动作进行部分观测，并保持梯度可传递
    :param act_logits: 动作的逻辑值
    :param ob_batch: 状态值
    :precursor_list: 前驱值
    :return: act
    """
    action_list = []
    for act_logit, ob, precursor in zip(act_logits, ob_batch, precursor_list):
        neighbors,neighbor_hot = ob.get_neighbors()
        # if precursor == 12:
        #     print(ob.get_neighbors())
        # print('前驱节点检查')
        # print(f'邻居节点{neighbors}')
        # print(f'前驱节点{precursor}')
        # print('#'*20)
        # assert precursor in neighbors or precursor == None, f'前驱节点{precursor}不在邻居节点{neighbors}中，请检查bug！'

        for node_index, node in enumerate(ob.node_ids):
            if node == precursor:
                neighbor_hot[node_index] = 0

        # 动作修正
        for next_hop in neighbors:
            if len(list(ob.G.neighbors(next_hop))) == 1 and ob.node_id in list(
                    ob.G.neighbors(next_hop)) and setting.action_modification:
                neighbor_hot[ob.node_ids == next_hop] = 0
                # 动作修正测试 -- 通过
                # print('动作修正')

        act_local = torch.unsqueeze(act_logit[0,neighbor_hot],dim=0)
        # 通过epsilon-贪心算法获取动作
        if explore:
            action = gumbel_softmax_batch(act_local)  # 可以进行梯度回传
        else:
            action = onehot_from_logits_batch(act_local)  # 不可以进行梯度回传

        action_board = torch.zeros([len(neighbor_hot), 1], dtype=act_logits.dtype, device=act_logits.device)
        action_board[neighbor_hot, 0] = action.view(-1)
        action_list.append(action_board)
    action_tensor = torch.stack(action_list,dim=0)
    return action_tensor


# 小批量进行训练
def generate_critic_input_batch(obs_batch,acts_batch,device,shuffle=True):
    data_list = []
    batch_size = len(obs_batch)
    for obs,acts in zip(obs_batch,acts_batch):
        if 'x' in locals(): del x
        for ob,act in zip(obs,acts):
            if not torch.is_tensor(act):
                act = torch.tensor(act)
            ind_agent = torch.cat((ob.to_Data().features.to(device), act), dim=1)
            if 'x' not in locals():
                x = ind_agent
            else:
                x = torch.cat((x, ind_agent[:,2:]), dim=1)
            adjacency = ob.to_Data().adjacency.to(device)
            if 'node_ids' not in locals():
                node_ids = ob.G.nodes
        data = setting.Data(features=x,adjacency=adjacency)
        data_list.append(data)
    loader = DataLoader(data_list,batch_size=batch_size,shuffle=shuffle)
    return loader
    # if 'features' not in locals():
    #     features = x.unsqueeze(0)
    # else:
    #     features = torch.cat((features, x.unsqueeze(0)), dim=0)
    # if 'adjacencys' not in locals():
    #     adjacencys = adjacency.unsqueeze(0)
    # else:
    #     adjacencys = torch.cat((adjacencys, adjacency.unsqueeze(0)), dim=0)

# 小批量获取动作
def generate_actor_input_batch(obs_batch,device,shuffle=True):
    """
    生成行动者网络的输入
    :param obs_batch: 批量状态状态
    :return:
    """
    data_list = []
    batch_size = len(obs_batch)
    for ob in obs_batch:
        features = ob.to_Data().features

        adjacency = ob.to_Data().adjacency.to(device)
        data = setting.Data(features=features, adjacency=adjacency)
        data_list.append(data)
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)
    return loader


# # 3. 使用MADDPG训练好的模型获取下一步，模型的应用
# def MADDPG_find_next_hop(G,current_hop,std):


if __name__ == '__main__':
    # Gdata = torch.load('gdata.pt')
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = Actor(Gdata,1,2,0.0001,device)
    # model.supervision_mask()
    with open('samples.pkl','rb') as f:
        sample = pickle.load(f)
    states, actions, rewards, next_states, done = sample
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('状态', states)
    print('动作', actions)
    print('收益', rewards)
    print('下一个动作', next_states)
    print('结束状态', done)
    # adjacency,feature = generate_critic_input(states,actions,device)
    # print('属性：', feature, feature.size())
    # print('邻接矩阵', adjacency, adjacency.size())
    # # 修改algorithm.py下类GNNCritic，使其适应Data类型的输入
    # # 智能体节点编号
    # agents_node = [1, 2, 4, 5, 6]
    # agents_num = len(agents_node)
    # state_dim = 4
    # critic_input_dim = (state_dim + 1) * agents_num
    # critic_hidden_dim = agents_num
    # actor_lr = 0.0001
    # critic_lr = 0.000001

    # node_id = 1
    # agents = DDPG(node_id, state_dim, critic_input_dim, critic_hidden_dim, actor_lr, critic_lr, device)
    # data = setting.Data
    # data(x=feature,adjacency=adjacency,node_ids=states[0].G.nodes)
    # value = agents.actor_evaluation(data)
    # print(value)
    #
    # gamma = 0.95
    # tau = 1e-2
    print(states)
    print(actions)
    loader = generate_critic_input_batch(states,actions,device)
    agents_node = [1, 2, 4, 5, 6]
    agents_num = len(agents_node)
    state_dim = 4
    critic_input_dim = (state_dim + 1) * agents_num
    critic_hidden_dim = agents_num
    actor_lr = 0.0001
    critic_lr = 0.000001
    node_id = 1
    node_ids = [1,2,4,5,6]
    agent = DDPG(node_id, state_dim, critic_input_dim, critic_hidden_dim,actor_lr, critic_lr, device)
    agents = []
    for node_id in node_ids:
        agents.append(DDPG(node_id, state_dim, critic_input_dim, critic_hidden_dim,actor_lr, critic_lr, device))
    # print(loader)
    for batch in loader:
        print(batch)
        print(batch.features)
        print(batch.features.size())

        value = agent.critic(batch,device)
        print(value)
        # print(value.size())


    for agt,next_obs in zip(agents,next_states.T):
         print(next_obs)
         actor_loader = generate_actor_input_batch(next_obs, device)
         for batch in actor_loader:
            act = agt.actor(batch,device)
            print(act)


    # for batch in actor_loader:
    #     act = agents.actor(batch,device)
    #     print(act)


    # TODO: 重新规划模型的输入数据类型为data，其中包含feature，adjacency两个特征

    # TODO：20240415  星期一
    # TODO：重写algorithms.py下的MADDPG下的update方法，使其适应批量训练
    # TODO: 修改algorithms.py下的partial_observation_batch函数，使其适应批量训练
    # TODO：为了适应批量处理，修改了algorithms.py下MADDPG类中的update函数。
    # TODO：同时，将algorithms.py下generate_actor_input函数扩展成可批量处理的generate_actor_input_batch函数。
    # TODO：将algorithms.py下generate_critic_input函数扩展成可批量处理的generate_critic_input_batch函数。
    # TODO：将algorithms.py下partial_observation函数扩展成可批量处理的partial_observation_batch函数。
    # TODO：将algorithms.py下GCNActor网络扩展成可批量处理的GCNActorBatch网络。
    # TODO：将algorithms.py下GNNCritic网络扩展成可批量处理的GNNCriticBatch网络。
    # TODO：修改了algorithms.py下GraphConvolution类中的forward方法，使其适应批量处理。







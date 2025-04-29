clc,clear
clear all;
crt = pwd;
addpath(genpath(crt));    % 添加当前的文件夹


% 读取相对路径下的数据文件
filepath = '../result/data/main_v130_data.mat';


reward_data = load(filepath);   % 多智能强化学习的奖励数据

num_list = reward_data.main_v130_num_list;    % 迭代周期
reward_value = reward_data.main_v130_evaluate_value; % 各智能体的奖励值
plot(num_list,reward_value)
xticks(0:300:1500);
xticklabels({'0','300','600','900','1200','1500'})

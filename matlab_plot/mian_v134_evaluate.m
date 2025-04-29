clc,clear
clear all;
crt = pwd;
addpath(genpath(crt));    % 添加当前的文件夹


% 读取相对路径下的数据文件
filepath = '../result/data/v134_model_evaluate.mat';
filepath2 = '../result/data/v130_model_evaluate.mat';

reward_data = load(filepath);   % 多智能强化学习的奖励数据
reward_data2 = load(filepath2);   % 多智能强化学习的奖励数据

num_list = reward_data.v134_episodes_list;    % 迭代周期
num_list2 = reward_data2.v130_episodes_list;    % 迭代周期


reward_value = reward_data.model_evaluate; % 各智能体的奖励值
reward_value2 = reward_data2.model_evaluate; % 各智能体的奖励值


figure
plot(num_list,reward_value(1,:))
yline(227)
xticks(0:300:1500);
xticklabels({'0','300','600','900','1200','1500'})
hold on 
plot(num_list2,reward_value2(1,:))
hold off

% 求最小值，并显示最小值的位置
[min_delay,min_delay_index] = min(reward_value(1,:));
disp(['完成任务时延最小为:',num2str(min_delay)])
disp(['完成任务时延最小的位置为:',num2str(num_list(min_delay_index)),'相应的能耗为:',num2str(reward_value(2,min_delay_index))])


figure
plot(num_list,reward_value(2,:))
yline(682.7827166055378)
xticks(0:300:1500);
xticklabels({'0','300','600','900','1200','1500'})
hold on 
plot(num_list2,reward_value2(2,:))
hold off

[min_energy,min_energy_index] = min(reward_value(2,:));
disp(['完成任务能耗最小为:',num2str(min_energy)])
disp(['完成任务能耗最小的位置为',num2str(num_list(min_energy_index)),'相应的完成任务的时延为:',num2str(reward_value(1,min_energy_index))])




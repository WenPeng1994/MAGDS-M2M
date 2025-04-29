clc,clear
clear all;
crt = pwd;
addpath(genpath(crt));    % 添加当前的文件夹


% 读取相对路径下的数据文件
filepath = '../result/data/v130_model_evaluate.mat';


reward_data = load(filepath);   % 多智能强化学习的奖励数据

num_list = reward_data.v130_episodes_list;    % 迭代周期
reward_value = reward_data.model_evaluate; % 各智能体的奖励值
figure
plot(num_list,reward_value(1,:))
yline(227)
xticks(0:300:1500);
xticklabels({'0','300','600','900','1200','1500'})

% 求最小值，并显示最小值的位置
[min_delay,min_delay_index] = min(reward_value(1,:));
disp(['完成任务时延最小为:',num2str(min_delay)])
disp(['完成任务时延最小的位置为',num2str(num_list(min_delay_index))])


figure
plot(num_list,reward_value(2,:))
yline(682.7827166055378)
xticks(0:300:1500);
xticklabels({'0','300','600','900','1200','1500'})

[min_energy,min_energy_index] = min(reward_value(2,:));
disp(['完成任务能耗最小为:',num2str(min_energy)])
disp(['完成任务能耗最小的位置为',num2str(num_list(min_energy_index))])




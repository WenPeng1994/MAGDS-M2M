# 开发日志
## 20240328 增加日志说明  ## 
### 对之前的日志进行补记
#### 20240302任务
1. 实现流量矩阵(traffic matrix)的仿真  20240302
2. 实现链路上经过数据量的统计   20240303
3. 实现剩余能量统计 
能量使用统计，每个节点发送一个数据包能量为105mW，接收一个数据包能量为54mW，模型待机30mW 20240304
4. 实现流量负载率的计算
链路的流量负载除以总流量
5. 实现链路丢包率的计算，丢包后重新发送，发送的能量有损耗，接收的能量无损耗 20240305


#### 20240304任务
设想对比实验
1. MAPPO
2. SPR：最短路径路由
3. Q-routing[17]
4. QELAR[20]
5. DQRC[33]
实现了两种情况，一种室随机跳到下一个邻居节点


#### 20240310
1. 计算各种中心性看相应的结果 20240310
2. 设置一个比例将前几个重要的指标取出来作为重要节点 20240311


#### 20240319
1. 解决算法名称代入的问题
2. 选择强化学习算法


#### 20240321
3. GCN 20240321
4. 暂时仅考虑节点上的属性


#### 20240324
在data_trans.py这个文件中新增GetState这个类，用于更新图状态的改变，包括以下三个部分：
1. 全局信息
2. 局部信息
3. 行为信息


#### 20240325
将动作转化为邻居节点的形式，将非邻居节点设置为0


#### 20240326
对全局性能进行归一化


#### 20240327
将仿真环境改写为强化学习环境


#### 20240328
1. 在logging文件夹下建立logging.md,用于记录开发日志
2. 补充完现有代码的日志
3. 新增MARL_env.py, 除了使用之前的类Camera，Switch，DataCenter，Topo外，新增PacketRoutingEnv这个类作为多智能体的环境。


#### 20240329
1. 更改了一下data_trans.py中DataTrans类下的up_local方法,将属性'src_std'改为‘std’，将之前的起点为'-1'，终点为'1',改成了只有终点为‘1’。
2. 在PacketRoutingEnv中新建step方法，用于获取智能体作用下的下一个状态和奖励函数，明天接着写。
3. 将当前节点记录到状态中，方便求邻居节点给出动作，修改了一下data_trans.py中DataTrans类下的up_local方法(如果状态可以直接获取邻居节点，可以不用标识当前节点)
4. 修改了模型(algorithms中DDPG)的take_action，使其实现部分观测，并且梯度可传递。
下一步任务：
a. 继续写PacketRoutingEnv中的step方法，获取智能体作用下的下一个状态和奖励函数。
b. 建立comments模块，将一些常用的模块和函数放入其中utils.py中，比如random_simulation函数。


#### 20240330 星期六
1. 继续写PacketRoutingEnv中的step方法，获取智能体作用下的下一个状态和奖励函数。
2. 建立了comments模块，将random_simulation函数放入其中utils.py中，用于判断是否传输成功。
3. 设计了成本计算，通过当前的状态快照计算成本。
4. 通过传输包更新全局状态。
5. 设计了能力管理下多次传输机制。


#### 20240331 星期日
1. 发现将pop()和更新分开，会出现问题。
2. 为了解决上述的问题，在Camera，Switch和DataCenter三个对象中添加观测observe的功能。
3. 将观测和传输分开，这样可以很方便进行操作。
4. 和胡洪文讨论了模型的正确性，以及可实现性。
5. 和蒋志邦讨论了图神经网络输入的拼接问题。
下一步任务：
a. 在PacketRoutingEnv中step中添加传输机制。


#### 20240401 星期一
1. 加入传输机制
2. 在PacketRoutingEnv下建立一个tran_data方法，用于传输数据
3. 在PacketRoutingEnv下建立一个reset方法，用于将状态
4. 新加数据包结构，重写封装.popleft()功能，简化获取数据包的方法
下一步任务：
a. 调试环境代码


#### 20240402 星期二
1. 修改data_trans.py里Get_state类中normalization方法，避免出现分母为零的状态。
2. 邻接矩阵需要转化为稀疏矩阵，并进行重排。这样比较麻烦，直接写成稠密矩阵的模式。
3. 修改了algorithms.py下DDPG中take_action的输出，这里可能为后面带来问题。
4. 修改了algorithms.py下random_short_path_hop函数，使得它可以适应没有数据的形式。
下一步的任务：
a. 独热编码中存在多个编码的形式，需要进一步调整。


#### 20240403 星期三
1. 修改agorithms.py下DDPG类中onehot_from_logits方法，用于处理最大动作不仅仅只有一个的情况，倾向于选择后面一个。
2. 在MARL_env.py中Camera类中添加add方法，用于处理数据发送失败的情况。
3. 修改algorithms.py 中类MADDPG中的update方法
下一步任务：
a. 继续写MADDPG中的update方法
    1. 获取目标动作
    2. 构建一个函数，将状态和动作组成评价网络的输入。
    3. 使用评价网络和目标评价网络获得值和目标值
    4. 通过奖励值更新评价网络
    5. 通过值更新行动者网络
    6. 利用软更新更新目标网络
b. 经过测试，随着能量的累积，成本越来越高，这似乎是不合理的，进一步修改成本函数。
c. 该测试只是在非探索的情况下使用的，如果是探索，还需进一步进行测试。


#### 20240405 星期五
1. 在data_trans.py中GetState类中添加__getstate__和__setstate__方法，确保可以存储。
2. 修改MARL_env.py中PacketRoutingEnv类中的step方法下的输出，让输出中包含状态。
3. 在setting中删除了algorithm的调用，排除了algorithm和setting循环调用的bug。
4. 在algorithms.py中写了一个函数generate_critic_input,将状态和动作组成评价网络的输入。
5. 在algorithms.py中写了一个函数generate_actor_input，用于将单个状态转化为行动者网络的输入。
6. 完成了algorithms.py中类MADDPG内update方法的书写，并进行了一步更新的测试。
下一步任务：
a. 写完训练过程，进一步测试。
b. 分析测试结果，做进一步的优化。


#### 20240406 星期六
1. 完成训练过程
2. 在main.py中写一个函数evaluate，用于对学习的策略进行评估，此时不会进行探索。
3. 不收敛，可能原因没有目标，添加终端奖励5000。
4. 可能原因没到达终点，修改一下每代训练结束的条件，增加每代训练的次数。
5. 可能原因没有进行梯度传递。
下一步任务：
a. 进一步检查问题。
b. 排除bug。


#### 20240407 星期日
1. 实验发现，评价网络是更新的，但动动作网络没有更新，动作网络在更新时没有进行部分观测。
2. 在algorithms.py中写一个函数partial_observation实现部分观测.
3. 将gumbel采样的函数从algorithms.py中DDPG类中独立出来，放到comments下的utils.py中，供其他函数调用。
4. 设计一个奖励函数，使得算法收敛。
5. 在MARL_env.py中写一个函数reward_calculate()来计算奖励函数。
6. 将缓冲区空的情况的收益记为零，对于能量比较高的情况下仅仅记录第一次。
7. 排除发送空数据包的情况。
8. 不稳定，可能是更新尺度太大，改变学习率。
下一步任务：
a. 换用不同的奖励函数，查看收敛情况


#### 20240408 星期一
1. 调整奖励函数，进行测试，修改了MARL_env.py中函数reward_calculate()函数。
2. 尝试过的奖励函数，第一，包含路径，局部数据包和局部累积使用的能量三个部分。第二，仅仅局部包含数据包和局部累积使用的能量两个部分。
   第三，全局数据包的总数。第四，全局数据包总数和全局累积累积使用的能量数。


#### 20240409 星期二
1. 在文件夹comments中utils.py中添加一个log_management函数，用于管理运行日志。
2. 在main.py中写一个函数convergence_result_display, 用于绘制收敛图。


#### 20240410 星期三
1. 尝试不同的奖励函数，查看收敛情况。
2. 准备使用批量训练的方法(先备份一下现在的代码)


#### 20240411 星期四
1. 进一步调试代码
2. 为使用批量训练做准备
   a. 获取到的样本有自己的数据类型，需要对其做处理。
   b. 需要对模型进行测试


#### 20240414 星期日
1. 修改data_trans.py下的类GetState下的to_Data函数，将x的命名改为features,去掉node_ids这个分量。
2. 规范一下数据格式。
3. 进行批量测试并修改。


#### 20240415 星期一
1. 为了适应批量处理，修改了algorithms.py下MADDPG类中的update函数。
2. 同时，将algorithms.py下generate_actor_input函数扩展成可批量处理的generate_actor_input_batch函数。
3. 将algorithms.py下generate_critic_input函数扩展成可批量处理的generate_critic_input_batch函数。
4. 将algorithms.py下partial_observation函数扩展成可批量处理的partial_observation_batch函数。
5. 将algorithms.py下GCNActor网络扩展成可批量处理的GCNActorBatch网络。
6. 将algorithms.py下GNNCritic网络扩展成可批量处理的GNNCriticBatch网络。
7. 修改了algorithms.py下GraphConvolution类中的forward方法，使其适应批量处理。


#### 20240416 星期二
1. 经过测试代码，需要进一步完善，代码从某一代重新开始训练。
2. 将训练结果部署到实际场景中，观察其运行结果。


#### 20240417 星期三
经过测试，程序收敛时跳跃过大，做以下改进
1. 将utils.py中的onehot_from_logits函数扩展为onehot_from_logits_max，在测试情况下直接使用独热编码，不再使用epsilon算法
2. 测试时将终端奖励改为按信息量得分,在main.py中修改函数evaluate。
3. 修改MARL_env.py中类PacketRoutingEnv下的step方法，使其打印测试结果。


#### 20240418 星期四
经过测试，程序收敛时跳跃过大，在做以下调整
1. 之前的可变学习率未生效，进行调整，在到达理想解后每10步降低5%，直到到达最低学习率。
2. 在main.py中新增学习率调整参数，将函数evaluate扩展为可以迭代的类Evaluate，写到utils中。

#### 20240422 星期一
经过测试，程序对初值敏感度比较大，这实际上是因为在训练中，目标节点很少到达的缘故，将终端奖励融入到单步中。
1. 修改了MARL_env.py下的reward_calculate函数。
2. 修改了setting.py下的参数reward_type。
3. 修改了unit.py下类Evaluate下的evalute评价方法。
4. 在setting.py中设置参数pre_state,用于记录之前的节点上数据包的个数。
5. 修改了main.py使其适应探索未到达终点的形式。

#### 20240423 星期二
1. 做了相应的备份first_research_point_v1
2. 修改缓冲区，形成记忆训练。
3. 将任务量修改成300试一下结果。


#### 20240425 星期四
1. 自main_v1.py之后将终端奖励改为指数型
2. 在setting.py中添加unit_reward表示单位奖励
3. 无输出运行main_v1.py,端口为65130
4. 修改了main_v1.py中rews的计算
5. 修改了utils.py中类Evaluate中的evaluate方法
6. main_v2提高探索率0.01-->0.1,降低batch32，加大更新步数2->5
7. 无输入运行main_v2.py，任务ID：425
8. 降低任务进行到150，将训练长度和周期都变为300
9. 无输入运行main_v2.py，任务ID：1241
10. 保持actor_lr不变，critic_lr下调10倍，0.00002 * 0.1
11. 无输入运行main_v4.py，任务ID：1529
12. 保持actor_lr不变，critic_lr上调10倍，0.00002 * 10
13. 无输入运行main_v5.py，任务ID：1722
14. 备份一份文件用于本地测试。
15. 测试评价网络是否更新。
16. 保持critic_lr=0.00002，actor_lr = 0.002*0.1
17. 无输入运行main_v6.py，任务ID：8061
18. 保持critic_lr=0.00002，actor_lr = 0.002*10
19. 无输入运行main_v7.py，任务ID：8191
20. 保持critic_lr=0.00002，actor_lr = 0.002*0.01, actor_lr==critic_lr
21. 无输入运行main_v8.py，任务ID：8331

#### 20240426 星期五
1. 保持critic_lr=0.00002，actor_lr = 0.002*0.001, actor_lr<critic_lr
2. 无输出运行main_v9.py，任务ID：12970
3. 保持critic_lr=0.00002，actor_lr = 0.002,任务数300，更新10，探索率0.1
4. 无输出运行main_v10.py，任务ID：1801820


#### 20240427 星期六
1. main_v11.py后将所有参数修改全部放入main中
2. 修改utils.py中的log_management函数
3. 在first_research_point - bt中进行尝试
4. 尝试简化模型，将actor变为一层卷积的形式，修改algorithms.py中的GCNActorBatch类
5. 将critic的输入变为全局状态加动作形式其维度为  智能体数*3+2，修改algorithms.py中的generate_critic_input函数

#### 20240428 星期日
1.对比结果，固定评论者的学习率0.00002，将行动者的学习率遍历[0.02，0.002,0.0002,0.00002，0.000002],从实验结果看0.002最好。补做0.02的实验。
2.降低batchsize
3.实验结果：0.002收敛，0.02，0.0002，0.00002，0.000002均不收敛


#### 20240429 星期一
1. 保存不同的模型，200代后每5代保存一次。
2. 修改了utils.py中的类Evaluate中的evaluate方法。
3. 修改了主函数中的信息打印和最后的学习率输出，参数不变跑一次。
4. 在actor_lr=0.002,critic_lr=0.00002的情况下对比的不同的batchsize，[2,4,8,16,32]，其中batchsize=2的情况下不稳定，
   batchsize=4的时候已经收敛。

#### 20240430 星期二
1. 下一步实验，将实验结果应用到仿真中。
2. 对比SPR(最短路径算法)。
3. 对比后设计相关指标，查看效果，进一步细调参数。

#### 20240502 星期四
1. 写算法应用，在utils.py中类Evalute下添加algorithm_application方法，用于算法评价。
2. 算法评价的应该包含各种算法的指标，节点上待处理的数据包变化情况，完成时间的计算。
3. 另外要统计每条边上经过的数据量。每个节点上能量的使用量。
4. 如果是强化学习算法，还得绘制收敛曲线。


#### 20240504 星期六
1. 修改MARL_env.py中的PacketRoutingEnv中的step的bug，用于规避索引过程中智能体没用的问题。
2. 在此基础上将main_v17.py作为main_v20.py，看看结果有什么变化。
3. 在MARL_env.py中的PacketRoutingEnv中添加新的属性link_msg_dict,用于统计链路上通过的数据包数,并添加了初始化。
4. 在MARL_env.py中tran_data函数中添加边上通过数据包数的逻辑。

#### 20240505 星期日
1. 新建一个文件alg_test_v1.py，用于算法部署.
2. 在MARL_env.py中的类PacketRoutingEnv中添加MARL_test_step方法，用于算法部署。

#### 20240506 星期一
1. 完成alg_test_v1.py的书写，统计节点数据包变化数，节点上能量使用数以及链路上数据包的通过量。
2. 在MARL_env.py中的类PacketRoutingEnv中添加othor_alg_step方法，用于传统算法部署。
3. 在alg_test_v1.py中添加node_package_display,node_energy_display和link_msg_display函数，用于绘制相应的性能图。


#### 20240507 星期二
1. 失眠的时候想到，实验效果低于随机最短路径算法的原因是因为受到7号交换机的最短随机路径的影响。


#### 20240508 星期三
1. 在utils.py中添加函数random_one_in_list，用于动作中具有多个1时，随机选择一个。
2. 修改utils.py中的onehot_from_logits函数，使其适应动作中出现多个为1的情况，类似地，修改onehot_from_logits_batch，onehot_from_logits_max函数。
3. 增加两种奖励函数，第9和第10，分别作为对照和实验，实验效果表明10的效果较好，高于随机最短路径算法。


#### 20240509 星期四
1. 添加任务的随机性，增加模型的适应性。
   1. 在MARL_env.py中添加Camera中添加打乱buffer中数据的方法buffer_resort。
   2. 修改MARL_env.py中的Topo类中的data_init方法。
2. 添加高能力采样的随机性
   因为样本是随机的，所以谁都可以作为第一个数据包，不影响采样的随机性。


#### 20240511 星期六
1. 对初值敏感性的测试，将随机种子改为0。


#### 20240515 星期三
1. 修正终端奖励处理错误的bug，在main47中进行测试。(最优版本)


#### 20240519 星期日
1. 在main48中加入重要性节点选择的算法。
2. 记录后10代结果的平均值用于评价，记录相关的数据。


#### 20240524 星期五
1. 新建node_select_env.py用于重要性节点的Q-learing选择
2. 在node_select_env.py添加Node_importance_env.py 作为重要性节点选择的环境
3. 修改algorithms.py中的rand_short_path_hop函数，通过参数importance_statistic控制是否进行节点重要性统计。
4. 在MARL_env.py中的类PacketRoutingEnv中添加node_importance_step方法来实现数据流向和节点重要性统计。


#### 20240525 星期六
1. 修改了node_select_env.py中类Node_importance_env的方法step,reward_calculation


#### 20240528 星期二
1. 修改MARL_env.py中的类PacketRoutingEnv中添加node_importance_step方法，重新计算
2. 修改node_select_env.py中Node_importance_env类中step方法，修正的奖励机制。
3. 修改MARL_env.py中的类PacketRoutingEnv中添加node_importance_step方法，修正奖励机制。


#### 20240605 星期三
1. 在重要性节点选择中添加了探索率衰减机制，修改了node_select_env.py中的主函数。
2. 在node_decision_number_display函数中添加了探索标记机制。
3. 写一个评价函数evaluation_function，一个重要性节点个数，跑10次求平均值。
4. 在MARL_env.py中的类PacketRoutingEnv中添加state_importance_step方法来实现数据流向和状态重要性统计。
5. 在MARL_env.py中的类PacketRoutingEnv中添加node_importance_epsilon_step方法来实现数据流向和节点重要性统计,实现重要性节点上用epsilon贪心策略采样.


#### 20240606 星期四
1. 在MARL_env.py中类PacketRoutingEnv中的方法node_importance_epsilon_step下添加新的奖励计算逻辑，保证可以收敛。
2. 在主函数V50中添加了全部保存后面几代的逻辑。


#### 20240611 星期二
去回路的操作
1. 在信息中添加上一步的信息
2. 修改相关的函数

#### 20240615 星期六
1. 为降低噪声的比重，修改utilsz中的 gumbel_softmax_sample函数 运行函数v70


#### 20240616 星期日
1. 将每个数据包的前驱加入到状态中。修改data_trans.py下的类GetState下的up_local方法。
2. main_v76, 终端奖励6000，actor网络学习率：0.002，批量16。
3. 基准实验：77：actor网络学习率：0.002，批量8。
4. 对比实验：78：actor网络学习率：0.0002，批量8。
5. 对比实验：79：actor网络学习率：0.02，批量8。

#### 忘了记的标志性事件
之前的拓扑中5连9，改成了5连10。

#### 20240617 星期一
1. 将奖励函数变成考虑下一个节点的能量。
2. 将初次生成的数据包变为前驱是None。
3. main_v80作为基准实验。
4. 对比实验 main_v81，actor学习率：0.0002
5. 对比实验 main_v82, actor学习率：0.02

#### 20240618 星期二
1. 将奖励函数变为两点的惩罚和
2. main_v84做为基准实验, a学习率：0.002，batch_size:8，minsize=4000
3. 对比实验 main_v83, a学习率：0.02, batch_size:8,minsize=40            收敛，但效果差
4. 对比实验 main_v85, a学习率：0.0002，batch_size:8，minsize=4000
5. 对比实验 main_v86, a学习率：0.002， batch_size:16,minsize=4000
6. 修改了邻接矩阵归一化的方式，重新运行基准实验main_v84
7. 再做一个对照实验， minisize = 2000
8. 再加一个对照实验，a学习率：0.02，batch_size:8,minisize=4000

#### 20240619 星期三
1. 基准实验main_90；a学习率：0.02，batch_size:8,minisize = 40      收敛
2. 补充对比实验main_91；a学习率：0.002，batch_size:8,minisize=40     不收敛
3. 补充对比实验main_92；a学习率：0.2，batch_size:8,minisize=40     收敛，波动大 0.1
4. 补充对比实验main_93；a学习率：0.02，batch_size:16,minisize=40    目前没收敛
5. 补充对比实验main_94；a学习率：0.02，batch_size:2,minisize=40     不收敛
6. 补充对比实验main_95；a学习率：0.02，batch_size:8,minisize=200    不收敛

#### 20240620 星期四
1. 补充实验main_96；a学习率：0.02，batch_size:4, minisize = 40   目前不收敛，不收敛
2. 补充实验main_97; a学习率：0.02, batch_size:6, minisize = 40   
3. 补充实验main_98; a学习率：0.02，batch_size:10, minisize = 40
4. 补充实验main_99; a学习率：0.02，batch_size:8, minisize = 80
5. 补充实验main_100; a学习率：0.02，batch_size:8, minisize = 20

#### 20240621 星期五
1. 修改了utils中Evaluate类中evaluate方法，将评价系统中去掉回路。
2. 基准实验main_101: a学习率：0.002，c学习率0.00002,batch_size:8,minisize:4000   # 可以到达终点，不持续
3. 对比实验main_102: a学习率：0.0002，c学习率0.00002,batch_size:8,minisize:4000   # 可以到达终点，不持续
4. 对比实验main_103: a学习率：0.02，c学习率0.00002,batch_size:8,minisize:4000     # 可以到达终点，不持续
5. 对比实验main_104: a学习率：0.002，c学习率0.00002,batch_size:4,minisize:4000
6. 对比实验main_105: a学习率：0.002，c学习率0.00002,batch_size:16,minisize:4000    # 收敛，但受回路影响特别严重


#### 20240623 星期日
1. 基准实验main_105: a学习率：0.002，c学习率0.00002,batch_size:16,minisize:4000, beta = 0.001     # 收敛，但受回路影响特别严重
2. 对比实验main_106: a学习率：0.002，c学习率0.00002,batch_size:16,minisize:4000, beta = 0.01
增加参数beta的显示 修改main函数
3. 对比实验main_107: a学习率：0.002，c学习率0.00002,batch_size:16,minisize:4000, beta = 0.1
4. 对比实验main_108: a学习率：0.002，c学习率0.00002,batch_size:16,minisize:4000, beta = 0.5
在utils中添加一个功能性函数tuple_t,用于将m×n的元组列表转化为n×m的元组列表--这是一个bug，导致前面的结果都是有问题的，这里需要重新运行101-105


#### 20240624 星期一
1. 基准实验main_101: a学习率：0.002，c学习率0.00002,batch_size:8,minisize:4000   # 可以到达终点，不持续
2. 对比实验main_102: a学习率：0.0002，c学习率0.00002,batch_size:8,minisize:4000   # 收敛  294slot 1148.2216  1-7 1-12 回路明显
3. 对比实验main_103: a学习率：0.02，c学习率0.00002,batch_size:8,minisize:4000     # 收敛  500slot 6400.2119 5-6 2-13 回路严重
4. 对比实验main_104: a学习率：0.002，c学习率0.00002,batch_size:4,minisize:4000     # 能到达终点，波动大
5. 对比实验main_105: a学习率：0.002，c学习率0.00002,batch_size:16,minisize:4000    # 收敛，500slot 6385.9900 2-13 回路严重

* 修正了初始前驱节点的bug，发现存在以下的回路没法避免，4(智能节点)->3(非智能节点)->4(唯一路径)
* 如果奖励存在正值，这样的回路就会不断出现，要想避免这样的回路，奖励必须是负的。
1. 基准实验main_106: a学习率：0.002，c学习率0.00002,batch_size:16,minisize:4000, beta = 0.001
2. 对比实验main_107: a学习率：0.002，c学习率0.00002,batch_size:16,minisize:4000, beta = 0.01
3. 对比实验main_108: a学习率：0.002，c学习率0.00002,batch_size:16,minisize:4000, beta = 0.1
4. 对比实验main_109: a学习率：0.002，c学习率0.00002,batch_size:16,minisize:4000, beta = 0.5

* 在每次结果后面添加最终剩余的数据包数 main_110
* 修复目标导向没生效的bug。
基准实验main_110: a学习率：0.002，c学习率0.00002,batch_size:16,minisize:4000, beta = 0.001

* 排除alg_test_v1中的问题。

#### 20240625 星期二
1. 修改utils中Evaluate类下的evaluate函数
2. 去除目标导向的逻辑，目标导向是基于轨迹的训练，这里适合使用。
3. 基准实验：main_111: a学习率：0.002，c学习率0.00002,batch_size:8,minisize:4000, beta = 1
4. 修复采样中的bug，多能力的节点上样本中前驱节点不对。
5. 基准实验：main_112: a学习率：0.002，c学习率0.00002,batch_size:8,minisize:4000, beta = 1

#### 20240626 星期三
1. 修改了评价函数中方法MARL_test_step下的bug.
2. 感觉受到终端奖励的影响，结果反而没法到达想要的结果，将终端奖励置为0看一下进行训练。
3. 基准实验：main_113: a学习率：0.002，c学习率0.00002,batch_size:8,minisize:4000, beta = 1，terminal_reward = 0
4. 对比实验: main_114: a学习率：0.002，c学习率0.00002,batch_size:8,minisize:4000, beta = 1，terminal_reward = 0, 学习率不变
5. 改变拓扑5-9导致程序全部停止，重新运行main_114
6. 基准实验: main_114: a学习率：0.002，c学习率0.00002,batch_size:8,minisize:4000, beta = 1，terminal_reward = 0, 学习率不变

#### 20240627 星期四
1. 基准实验: main_115: a学习率：0.002，c学习率0.00002,batch_size:8,minisize:4000, beta = 0.9，terminal_reward = 0, 学习率到达终点后每10代衰减0.02，奖励机制单点制奖励类型10.
2. 修订奖励10的bug,把数据包数量当能量数量使用，main_v114不受影响。
3. 在main.py中添加python转matlab数据形式。
4. 基准实验: main_116: a学习率：0.002，c学习率0.00002,batch_size:8,minisize:4000, beta = 0.9，terminal_reward = 0, 学习率到达终点后每5代衰减0.02，奖励机制单点制奖励类型10. 运行周期500
5. 对比实验: main_117: a学习率：0.002，c学习率0.00002,batch_size:8,minisize:4000, beta = 0.95，terminal_reward = 0, 学习率到达终点后每5代衰减0.02，奖励机制单点制奖励类型10. 运行周期500

#### 20240628 星期五
1. 基准实验: main_118: a学习率：0.002，c学习率0.00002,batch_size:8,minisize:4000, beta = 1，terminal_reward = 0, 学习率到达终点后每5代衰减0.02，奖励机制单点制奖励类型10. 运行周期500
2. 对比实验: main_119: a学习率：0.002，c学习率0.00002,batch_size:8,minisize:4000, beta = 0.975，terminal_reward = 0, 学习率到达终点后每5代衰减0.02，奖励机制单点制奖励类型10. 运行周期500
3. 对比实验: main_120: a学习率：0.0002，c学习率0.00002,batch_size:8,minisize:4000, beta = 1，terminal_reward = 0, 学习率到达终点后每5代衰减0.02，奖励机制单点制奖励类型10. 运行周期500      # 效果更差


#### 20240629 星期六
1. 基准实验: main_121: a学习率：0.002，c学习率0.00002,batch_size:8,minisize:4000, beta = 1，terminal_reward = 0, 学习率到达终点后每5代衰减0.02，奖励机制单点制奖励类型10. 运行周期1000      # 趋势还在降，写一个继续训练的逻辑
   没见大波动，结果中出现了213slot,775.9019
2. 对比实验：main_122: a学习率：0.02，c学习率0.00002,batch_size:8,minisize:4000, beta = 1，terminal_reward = 0, 学习率到达终点后每5代衰减0.02，奖励机制单点制奖励类型10. 运行周期500   # 已见波动
3. 对比实验：main_123: a学习率：0.002，c学习率0.00002,batch_size:8,minisize:4000, beta = 0.99，terminal_reward = 0, 学习率到达终点后每5代衰减0.02，奖励机制单点制奖励类型10. 运行周期500  
   在300左右波动，但是没超过最短路径，结果略显差于main_v118.


4. 对比实验: main_124: a学习率：0.002，c学习率0.00002,batch_size:8,minisize:4000, beta = 0.9，terminal_reward = 0, 学习率到达终点后每5代衰减0.02，奖励机制单点制奖励类型11. 运行周期500

#### 20240630 星期日
1. 对比实验：main_125: a学习率：0.01，c学习率0.00002,batch_size:8,minisize:4000, beta = 1，terminal_reward = 0, 学习率到达终点后每5代衰减0.02，奖励机制单点制奖励类型10. 运行周期500  # 
   有波动，用时277slot,940.0695
2. 对比实验：main_126: a学习率：0.005，c学习率0.00002,batch_size:8,minisize:4000, beta = 1，terminal_reward = 0, 学习率到达终点后每5代衰减0.02，奖励机制单点制奖励类型10. 运行周期500  #
   大波动减小，用时271slot，891.518

增加继续训练逻辑training_continue函数(明天进行测试)
1. 在原有的缓冲区类中添加save和load方法，修改utils.py下ReplayBuffer类下的方法.
2. 修改main函数，使它可以保存buffer，main_v127版本之后。
3. 基准实验：main_v127 a学习率：0.002，c学习率0.00002,batch_size:8,minisize:4000, beta = 1，terminal_reward = 0, 学习率到达终点后每5代衰减0.02，奖励机制单点制奖励类型10. 运行周期1500
   异常停止
4. 修正main函数中学习率显示并保存的问题，main_v127版本之后生效。
注意:main_v127做了很多改变，后面的版本要以这个为主。


#### 20240701 星期一
1. 对比实验(测上限)：main_128: a学习率：0.003，c学习率0.00002,batch_size:8,minisize:4000, beta = 1，terminal_reward = 0, 学习率到达终点后每5代衰减0.02，奖励机制单点制奖励类型10. 运行周期500  #
开始的时候比较好，后面的结果稳定在270左右
2. 对比实验(测下限)：main_129: a学习率：0.001，c学习率0.00002,batch_size:8,minisize:4000, beta = 1，terminal_reward = 0, 学习率到达终点后每5代衰减0.02，奖励机制单点制奖励类型10. 运行周期500

#### 20240703 星期三
1. 基准实验：main_v130 a学习率：0.002，c学习率0.00002,batch_size:8,minisize:4000, beta = 1，terminal_reward = 0, 学习率到达终点后每5代衰减0.02，奖励机制单点制奖励类型10. 运行周期1500
   127重新运行

#### 20240705 星期五
1. 对比实验：main_v131 a学习率：0.002，c学习率0.00002, batch_size:8, minisize:4000, beta = 0.99，terminal_reward = 0, 学习率到达终点后每5代衰减0.02，奖励机制单点制奖励类型10. 运行周期1500
   
#### 20240711 星期四
1. 在setting中新增topo路径
2. 对比实验：main_v132 a学习率：0.002，c学习率0.00002, batch_size:8, minisize: 4000, beta = 0.99，terminal_reward = 0, 学习率到达终点后每5代衰减0.02，奖励机制单点制奖励类型12. 运行周期1500
3. 对比实验：main_v133 a学习率：0.002，c学习率0.00002, batch_size:8, minisize: 4000, beta = 0.99，terminal_reward = 0, 学习率到达终点后每5代衰减0.02，奖励机制单点制奖励类型10. 运行周期1500，考虑时延

#### 20240713 星期六
1. main_v133运行出错，启动main_v134


#### 20241223 星期一
1. 补做实验 main_v132_4_1.py 智能体[1,4]                        2
2. 补做实验 main_v132_4_2.py 智能体[1,4,5]                      3
3. 补做实验 main_v132_4_3.py 智能体[1,4,2,6]                    4
4. 补做实验 main_v132_4_4.py 智能体[1,2,4,5,11,15]              6
5. 补做实验 main_v132_4_5.py 智能体[1,2,4,5,6,10,11]  失败       7

#### 20241225 星期三
1. 补做实验 main_v132_4_6.py 智能体[1,2,4,5,6,10,11]
# README

1.训练模型

执行main_v*.py文件

2.评价模型

执行alg_test_v1.py文件

注意修改下面参数

alg_name = 'magddpg'  # 使用强化学习方法

alg_name = 'other_alg'  # 使用最短路径方法

model_evaluate_type = True  # 是否逐代进行评估，能看到每一代的结果

model_evaluate_type = False  # 只进行一代评估，能看到一代的结果，节点上的数据包变化情况等.

model_ver = "v*"  # 这里版本号对应希望评估main_v\*文件的版本，使用最短路径方法时可以忽略这个参数

model_eps = '*'    # 这里对应选取模型的代数，注意需要进行逐代评价生成相应的文件之后，才能有效打开对应的文件，此外还需注意逐代评价时有保存对应的模型。







 
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Auther:Evans_WP
@Contact:WenPeng19943251019@163.com
@Time:2025/2/15 9:34
@File:stochastic_node.py
@Desc:随机选择节点
"""
import random
# 设置可重复随机种子
seed = 1234
random.seed(seed)

# 利用随机选择获取节点
choice_node_numbers = 5
experiment_numbers = 20
node_list = [1,2,3,4,5,6,7,8,9,10,11,15]
for i in range(experiment_numbers):
    choice_node = random.sample(node_list,k=choice_node_numbers)
    print(sorted(choice_node))


"""
[1, 2, 8, 10, 11]   *
[2, 4, 6, 11, 15]   *
[1, 6, 8, 11, 15]
[2, 3, 8, 10, 15]
[1, 2, 8, 9, 15]
[2, 4, 8, 9, 11]
[1, 2, 9, 10, 11]
[2, 5, 8, 9, 15]
[4, 5, 6, 8, 10]   *
[1, 2, 9, 11, 15]
[1, 3, 6, 7, 8]
[1, 2, 3, 8, 15]
[1, 2, 3, 4, 9]
[2, 3, 5, 6, 9]
[1, 5, 7, 8, 11]
[3, 5, 10, 11, 15]  * 
[1, 6, 7, 8, 10]
[6, 8, 9, 10, 15]   *
[1, 3, 8, 10, 11]
[5, 8, 9, 11, 15]
"""





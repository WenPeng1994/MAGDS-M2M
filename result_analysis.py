#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Auther:Evans_WP
@Contact:WenPeng19943251019@163.com
@Time:2024/9/10 9:38
@File:result_analysis.py
@Desc:****************
"""
import numpy as np

time_slots = np.array([173,182,172,177,161])
print(-(time_slots-225)/225)

energy_std = np.array([673.391,793.828,728.645,826.911,677.60])
print(-(energy_std-740.76)/740.76)

time_slots = np.array([203,188,191,194,172])
print(-(time_slots-225)/225)

energy_std = np.array([589.34,599.69,601.76,578.76,622.27])
print(-(energy_std-740.76)/740.76)

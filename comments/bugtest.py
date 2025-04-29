#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Auther:Evans_WP
@Contact:WenPeng19943251019@163.com
@Time:2024/9/19 16:03
@File:bugtest.py
@Desc:****************
"""
# bug检测函数
# 检测变量
import textwrap
def test_variable(code_str,var):
    try:
        exec(textwrap.dedent(code_str))
    except:
        print(exec(textwrap.dedent(var)))
        assert f'出现了错误'



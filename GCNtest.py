#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Auther:Evans_WP
@Contact:WenPeng19943251019@163.com
@Time:2024/4/12 15:33
@File:GCNtest.py
@Desc:****************
"""
import torch
from torch.nn.modules import Module
from torch_geometric.nn import GCNConv


class GCN(Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return x


# 假设你有一个GCN层和待测试的数据
gcn_layer = GCN(16, 32)  # in_channels=16, out_channels=32
x = torch.randn(10, 16)  # 假设batch size为10, in_channels为16
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)

# 测试数据形状
print(x.shape)  # 应该输出 torch.Size([10, 16])
print(edge_index.shape)  # 应该输出 torch.Size([2, 4])

# 前向传播，如果形状不匹配会抛出异常
out = gcn_layer(x, edge_index)
print(out.shape)  # 输出GCN层处理后的输出形状
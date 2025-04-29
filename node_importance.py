#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Auther:Evans_WP
@Contact:WenPeng19943251019@163.com
@Time:2024/3/10 16:19
@File:node_importance.py
@Desc:****************
"""
import networkx as nx
import logging
import setting
import math

class NodeImportance:
    def __init__(self,G):
        self.G = G
        self.degree_centrality = self.degree_centrality()
        self.betweenness_centrality = self.betweenness_centrality()
        self.closeness_centrality = self.closeness_centrality()
        self.eigenvector_centrality = self.eigenvector_centrality()
        self.katz_centrality = self.katz_centrality()
        self.pagerank_centrality = self.pagerank_centrality()

    # 度中心性
    def degree_centrality(self):
        return nx.degree_centrality(self.G)

    # 介数中心性
    def betweenness_centrality(self):
        return nx.betweenness_centrality(self.G)

    # 接近中心性
    def closeness_centrality(self):
        return nx.closeness_centrality(self.G)

    # 特征向量中心性
    def eigenvector_centrality(self):
        return nx.eigenvector_centrality(self.G)

    # katz中心性
    def katz_centrality(self):
        return nx.katz_centrality(self.G)

    # PageRank中心性
    def pagerank_centrality(self):
        return nx.pagerank(self.G)

    def display(self):
        logging.info('度中心性:{}'.format(self.degree_centrality))
        logging.info('介数中心性:{}'.format(self.betweenness_centrality))
        logging.info('接近中心性:{}'.format(self.closeness_centrality))
        logging.info('特征向量中心性:{}'.format(self.eigenvector_centrality))
        logging.info('Katz中心性:{}'.format(self.katz_centrality))
        logging.info('PageRank中心性:{}'.format(self.pagerank_centrality))
        # print('度中心性',self.degree_centrality)
        # print('介数中心性',self.betweenness_centrality)
        # print('接近中心性',self.closeness_centrality)
        # print('特征向量中心性',self.eigenvector_centrality)
        # print('Katz中心性',self.katz_centrality)
        # print('PageRank中心性',self.pagerank_centrality)

    # 节点重要性分析
    def node_centrality_analyze(self):
        # 将节点按度中心排序
        # centrality = [self.degree_centrality,self.betweenness_centrality,self.closeness_centrality,
        #              self.eigenvector_centrality,self.katz_centrality,self.pagerank_centrality]
        # centrality_name = ['度中心性','介数中心性','接近中心性','特征向量中心性','Katz中心性','PageRank中心性']
        centrality = [self.degree_centrality, self.betweenness_centrality, self.closeness_centrality,
                      self.eigenvector_centrality]
        centrality_name = ['度中心性', '介数中心性', '接近中心性', '特征向量中心性']

        # 取出重要性节点
        proportion = setting.proportion  # 重要性节点选择概率
        critical_node_set = set()  # 重要性节点集合
        for name,name_dict in zip(centrality_name,centrality):
            # 对每一类中心性排序，并输出相应的结果
            name_dict_sorted = sorted(name_dict.items(),key=lambda x:x[1],reverse=True)
            order = [item[0] for item in name_dict_sorted]
            logging.info(f'{name}作用下的排列顺序{order}')
            critical_node_set.update(order[0:math.floor(proportion*len(order))])
        logging.info(f"重要性节点集:{critical_node_set}")
        return critical_node_set















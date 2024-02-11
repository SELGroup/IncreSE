# 生成三个随机过程得到的动态图数据集
# 包括一个初始图（nx.Graph），和20段增量序列，每段增量序列内的所有增量边没有先后顺序

import networkx as nx
import matplotlib.pyplot as plt
import community
import networkx.algorithms.community as nx_comm
from gensim.models import Word2Vec
from node2vec import Node2Vec
import numpy as np
import random
import os
import math
import time

class dataset_generator:
    def __init__(self, g = None):
        # nx.Graph结构的图，每个节点带一个d维的节点嵌入属性
        self.graph: nx.Graph = g
        self.node_vec = None

    def set_graph(self, graph):
        self.graph = graph

    # 从数据集中读取数据并构造出一个nx.Graph，赋值给self.graph，此时不带节点嵌入
    def init_graph(self, dataset):
        g = nx.karate_club_graph()
        self.graph = g

    # 人工构造一个nx.Graph对象，赋值给self.graph，此时不带节点嵌入
    def manmade_graph(self):
        # g = nx.karate_club_graph()
        # g = nx.powerlaw_cluster_graph(200,2,0.1)
        # g = nx.random_partition_graph([3, 3, 4], 0.3, 0.01, seed=12)
        # g = nx.random_partition_graph([20, 20, 10], 0.3, 0.01, seed=12)
        # g = nx.random_partition_graph([30, 30, 40], 0.3, 0.01, seed=12)
        # g = nx.random_partition_graph([200, 300, 200, 100, 200], 0.3, 0.01, seed=12)
        # g = nx.random_partition_graph([600, 600, 800], 0.3, 0.01, seed=12)
        g = nx.random_partition_graph([400, 600, 300, 200, 500], 0.3, 0.01, seed=12)
        self.graph = g

    # 人工构造一个随机划分图
    def make_random_partition_graph(self, nodenum_list, pin, pout, seed):
        g = nx.random_partition_graph(nodenum_list, pin, pout, seed=seed)
        self.graph = g

    def get_partition(self):
        part = community.best_partition(self.graph)
        dict = {}
        for node in part:
            comm = part[node]
            if comm in dict:
                dict[comm].append(node)
            else:
                dict[comm] = [node]
        # print(dict)
        # {comm1: [n1,n2,n3], comm2:[n4,n5,n6]}
        return dict

    # 按照最优划分来打印图
    def print_graph(self):
        partition = community.best_partition(self.graph)
        print(nx_comm.louvain_communities(self.graph))
        pos = nx.spring_layout(self.graph)
        plt.figure(figsize=(8, 8))
        plt.axis('off')
        nx.draw_networkx_nodes(self.graph, pos, node_size=100, cmap=plt.cm.RdYlBu, node_color=list(partition.values()))
        nx.draw_networkx_edges(self.graph, pos, alpha=0.3)
        plt.show()

    # 根据当前图随机产生规模为n的增量序列：有概率p在随机社区内加一条边，有概率q在随机社区间加一条边，有概率（1-p-q）在任意节点处新加一个节点
    def generate_seq_random(self, n, inner_prob, outer_prob, epoch = 10):
        part = self.get_partition()  # 获取Louvain产生的最优划分作为初始划分 {comm1: [n1,n2,n3], comm2:[n4,n5,n6]}
        comm_list = list(part.keys())  # [comm1, comm2]
        seq = []
        node_marker = len(self.graph.nodes)  # 自增的新节点编号器，编号类型为int，从节点列表长度数开始
        time_buffer = time.time()
        p = inner_prob
        q = outer_prob

        num_per_epoch = n / epoch

        for i in range(n):
            point = random.random()
            if point < p:
                while True:
                    target_comm = random.sample(comm_list, 1)[0]  # 随机选择一个社区
                    target_list = part[target_comm]
                    n1, n2 = random.sample(target_list, 2)
                    if (n1, n2) not in self.graph.edges:
                        self.graph.add_edge(n1, n2)
                        seq.append((n1, n2))
                        break
            elif p < point < p + q:
                while True:
                    c1, c2 = random.sample(comm_list, 2)
                    target_list1 = part[c1]
                    target_list2 = part[c2]
                    n1 = random.sample(target_list1, 1)[0]
                    n2 = random.sample(target_list2, 1)[0]
                    if (n1, n2) not in self.graph.edges:
                        self.graph.add_edge(n1, n2)
                        seq.append((n1, n2))
                        break
            else:
                target_comm = random.sample(comm_list, 1)[0]
                target_node = random.sample(part[target_comm], 1)[0]
                self.graph.add_edge(target_node, node_marker)
                seq.append((target_node, node_marker))
                part[target_comm].append(node_marker)
                node_marker += 1

            if i % num_per_epoch == 0:
                # print(f'epoch: {i / num_per_epoch}/{epoch} timestamp: {time.time() - time_buffer}')
                time_buffer = time.time()

        return seq

    # node2vec训练当前图的节点嵌入
    def node2vec(self, load=False):
        node2vec_model_save_path = 'node2vec_model'

        node_kv = None
        if os.path.exists(node2vec_model_save_path) and load:
            node_kv = Word2Vec.load(node2vec_model_save_path)
            print('loaded node2vec model')
        else:
            node2vec_model = Node2Vec(self.graph, workers=1, dimensions=16)  # 训练node2vec
            node_kv = node2vec_model.fit(window=10, min_count=1, batch_words=4)
            node_kv.save(node2vec_model_save_path)  # 保存模型

        self.node_vec = node_kv.wv

        # self.node_vec[0] = self.node_vec[1]
        # self.node_vec['v1'] = self.node_vec[1]
        # print(self.node_vec[0],self.node_vec[1])
        # print(self.node_vec['v1'])

    # node2vec训练当前图的节点嵌入，并进行存储
    def node2vec_save(self, save_path):
        node2vec_model = Node2Vec(self.graph, workers=1, dimensions=16)  # 训练node2vec
        node_kv = node2vec_model.fit(window=10, min_count=1, batch_words=4)
        node_kv.save(save_path)  # 保存模型
        self.node_vec = node_kv.wv

    # 根据当前图用霍克斯随机过程产生规模为n的增量序列
    def generate_seq_hawkes(self, n, p_edge, sample_num):

        seq = []
        node_marker = len(self.graph.nodes)
        node_list = self.graph.nodes
        p = p_edge  # 边概率，1-p的概率随机生成一个新节点

        for edge in self.graph.edges:
            self.graph.edges[edge]['time'] = 0

        # 返回两个节点的表示的欧几里得距离平方的相反数
        def f(n1, n2):
            return -np.linalg.norm(self.node_vec[n1] - self.node_vec[n2]) ** 2

        def softmax(list):
            arr = np.array(list)
            arr -= np.max(arr)
            arr = np.exp(arr) / np.sum(np.exp(arr))
            result_list = arr.tolist()
            return result_list

        # 分epoch次生成
        epoch = 5
        num_per_epoch = math.ceil(n / epoch)
        for i in range(epoch):
            epoch_start_time = time.time()
            current_time = i + 1
            self.node2vec()
            if (i + 1) * num_per_epoch > n:
                num_per_epoch = n - i * num_per_epoch

            for j in range(num_per_epoch):
                point = random.random()
                if point < p:  # 加边
                    target_node = random.sample(node_list, 1)[0]
                    # 条件强度函数序列
                    other_node_list = []
                    lambda_list = []
                    node_list_part = random.sample(node_list, sample_num)  # sample to speed up: math.ceil(len(node_list) / 40)
                    for other in node_list_part:
                        if other != target_node and (target_node, other) not in self.graph.edges:
                            current_lambda = f(target_node, other)
                            delta = self.graph.degree[target_node]  # 折扣率
                            for neighbour in self.graph.neighbors(target_node):
                                neighbour_time = self.graph.edges[(target_node, neighbour)]['time']
                                current_lambda += f(neighbour, other) * math.exp(
                                    -delta * (current_time - neighbour_time))
                            lambda_list.append(current_lambda)
                            other_node_list.append(other)
                    if other_node_list == []:
                        print(f'node {target_node} has no candidate neighbours')
                        break
                    prob_list = softmax(lambda_list)
                    arrived_neighbour = random.choices(other_node_list, weights=prob_list, k=1)[0]
                    self.graph.add_edge(target_node, arrived_neighbour)
                    self.graph.edges[(target_node, arrived_neighbour)]['time'] = current_time
                    seq.append((target_node, arrived_neighbour))
                else:  # 加节点
                    target_node = random.sample(node_list, 1)[0]
                    self.graph.add_edge(target_node, node_marker)
                    self.graph.edges[(target_node, node_marker)]['time'] = current_time
                    seq.append((target_node, node_marker))
                    self.node_vec[node_marker] = self.node_vec[target_node]
                    node_marker += 1
                # print('check:', i, j)
            epoch_end_time = time.time()
            print(f'epoch: {i}/{epoch} time cost: {epoch_end_time - epoch_start_time}')
        return seq

    # 根据当前图用三角闭合随机过程产生规模为n的增量序列
    def generate_seq_triad(self, n, p_edge):
        seq = []
        node_marker = len(self.graph.nodes)
        node_list = self.graph.nodes
        p = p_edge  # 边概率，1-p的概率随机生成一个新节点

        # 分epoch次生成
        epoch = 5
        num_per_epoch = math.ceil(n / epoch)
        for i in range(epoch):
            epoch_start_time = time.time()
            self.node2vec()
            if (i + 1) * num_per_epoch > n:
                num_per_epoch = n - i * num_per_epoch
            for j in range(num_per_epoch):
                point = random.random()
                if point < p:
                    while True:
                        target_node = random.sample(node_list, 1)[0]
                        if self.graph.degree[target_node] >= 2:
                            neighbour_list = list(self.graph.neighbors(target_node))
                            n1, n2 = random.sample(neighbour_list, 2)
                            if (n1,n2) in self.graph.edges:
                                continue
                            common_neighbors = list(nx.common_neighbors(self.graph, n1, n2))
                            theta = np.ones(16)  # 社会策略参数
                            prob_not_generate_edge = 1
                            for node in common_neighbors:
                                x = (self.node_vec[node] - self.node_vec[n2]) + (
                                        self.node_vec[node] - self.node_vec[n2])
                                prob = 1 / (1 + np.exp(- np.dot(theta, x)))
                                prob_not_generate_edge *= (1 - prob)
                            point2 = random.random()
                            if point2 < 1 - prob_not_generate_edge:
                                self.graph.add_edge(n1, n2)
                                seq.append((n1, n2))
                                break
                else:
                    target_node = random.sample(node_list, 1)[0]
                    self.graph.add_edge(target_node, node_marker)
                    seq.append((target_node, node_marker))
                    self.node_vec[node_marker] = self.node_vec[target_node]
                    node_marker += 1
            epoch_end_time = time.time()
            print(f'epoch: {i}/{epoch} time cost: {epoch_end_time - epoch_start_time}')
            print(len(seq))
        return seq


if __name__ == '__main__':
    gg = dataset_generator()
    gg.manmade_graph()
    gg.get_partition()
    example_seq = gg.generate_seq_random(50)
    print(example_seq)
    # gg.print_graph()

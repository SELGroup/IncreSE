# 增量图数据结构，用于基于给定调整（adjustment）进行结构熵增量计算
# Adjustment是一类时序数据，一个t时刻的Adjustment描述了t时刻图的增量数据（节点度、社区割边、社区容量的变化），以及t到t+1时刻的节点聚类结果变化
# 一个Adjustment需要依附于上下文环境 —— t时刻的聚类结果

import math
import time

import networkx as nx
from copy import *
from clusteror import Clusteror
from utils.statistic_tool import *
from utils.display_tools import *
from utils.example_graph import *


class IncreGraph:
    def __init__(self, graph: nx.Graph, clustering):
        self.graph = graph

        # node-comm mapping
        self.id_clustering = self.get_identified_clustering(clustering)  # 每个社区id对应的节点列表
        self.cluster_id = {}  # 每个节点id对应的社区id

        # Structural Data
        self.sd_node = {}  # {degree1: num1, degree2: num2, ...}
        self.sd_comm = {}  # {comm_id1: [cut1, vol1], comm_id2: [cut2, vol2], ...}
        self.sd_graph = {'cut': 0, 'vol': 0}  # {'cut': cut_num, 'vol': vol_num}

        # Structural Expression
        self.sexp_node = 0
        self.sexp_comm = 0
        self.sexp_graph = 0

        self.get_init_statistics()  # 进行初始化统计

    # 为每一个社区设置一个社区伪标签--id
    # 输入：聚类结果clustering，一个表示所有社区的嵌套列表
    # 输出：带id的聚类结果identified_clustering，一个字典，key是社区id，value是从属于key社区的节点列表
    def get_identified_clustering(self, clustering):
        id = 0
        identified_clustering = {}
        for cluster in clustering:
            identified_clustering[id] = cluster
            id += 1
        return identified_clustering

    # 初始化统计，给定一个图及其社区划分，从0开始统计各个层级的结构数据，并计算初始结构表达式的值：
    # 输入：带id的聚类结果identified_clustering
    # 输出：维护id映射，统计各个层级的结构数据，并计算初始结构表达式的值
    def get_init_statistics(self):
        # 统计结构数据
        for node in self.graph.nodes:
            degree = self.graph.degree[node]
            incre_update(self.sd_node, degree, 1)

        for id in self.id_clustering:
            cut = get_cut(self.graph, self.id_clustering[id])
            vol = get_volume(self.graph, self.id_clustering[id])
            self.sd_comm[id] = [cut, vol]
            self.sd_graph['cut'] += cut
            self.sd_graph['vol'] += vol

            # 维护id映射
            for node in self.id_clustering[id]:
                self.cluster_id[node] = id

        # 计算初始结构表达式
        for d in self.sd_node:
            k = self.sd_node[d]
            self.sexp_node += k * d * math.log2(d)

        for id in self.id_clustering:
            cut, vol = self.sd_comm[id]
            self.sexp_comm += (cut - vol) * math.log2(vol)
            self.sexp_graph += -cut

    # 基于adjustment进行的结构熵快速计算，并决定是否将该增量实际用于修改当前存储的结构数据和结构表达式
    def adjustment_based_SE_fast_computation(self, adjustment, update=False):
        sd_node_adj = adjustment[0]
        sd_comm_adj = adjustment[1]
        sd_graph_adj = adjustment[2]

        if update:
            # 计算结构表达式，更新结构数据和结构表达式
            for d in sd_node_adj:
                delta_k = sd_node_adj[d]
                self.sexp_node += delta_k * d * math.log2(d)
                incre_update(self.sd_node, d, delta_k)

            for id in sd_comm_adj:
                delta_cut, delta_vol = sd_comm_adj[id]
                cut, vol = self.sd_comm[id]
                self.sexp_comm -= (cut - vol) * math.log2(vol)

                if vol + delta_vol != 0:
                    self.sexp_comm += (cut + delta_cut - vol - delta_vol) * math.log2(vol + delta_vol)

                if id in self.sd_comm:
                    self.sd_comm[id][0] += delta_cut
                    self.sd_comm[id][1] += delta_vol
                else:
                    self.sd_comm[id][0] = delta_cut
                    self.sd_comm[id][1] = delta_vol

            self.sexp_graph += - sd_graph_adj['cut']
            self.sd_graph['cut'] += sd_graph_adj['cut']
            self.sd_graph['vol'] += sd_graph_adj['vol']

            # 计算更新后的结构熵
            updated_SE = -1 / self.sd_graph['vol'] * (
                    self.sexp_node + self.sexp_comm + self.sexp_graph * math.log2(self.sd_graph['vol']))

        else:
            sexp_node = self.sexp_node + 0
            sexp_comm = self.sexp_comm + 0
            sexp_graph = self.sexp_graph + 0

            # 计算结构表达式
            for d in sd_node_adj:
                delta_k = sd_node_adj[d]
                sexp_node += delta_k * d * math.log2(d)

            # print(sd_comm_adj)
            for id in sd_comm_adj:
                if id != 'None':
                    delta_cut, delta_vol = sd_comm_adj[id]
                    cut, vol = self.sd_comm[id]
                    sexp_comm -= (cut - vol) * math.log2(vol)
                    # print(id)
                    # print(vol + delta_vol)
                    if vol + delta_vol != 0:
                        sexp_comm += (cut + delta_cut - vol - delta_vol) * math.log2(vol + delta_vol)

            sexp_graph += - sd_graph_adj['cut']

            # 计算更新后的结构熵
            updated_SE = -1 / (self.sd_graph['vol'] + sd_graph_adj['vol']) * (
                    sexp_node + sexp_comm + sexp_graph * math.log2(self.sd_graph['vol'] + sd_graph_adj['vol']))

        return updated_SE

    # 生成一个节点偏移调整策略的adjustment，可支持减边、临时孤立边（增量内部无序）等
    def node_shifting_adjust(self, incre_seq, verbose=False, iter_num = 5):
        id_clustering_adj = {}  # 调整格式：{社区id：[(变动的节点，‘+/-’),(),()]}
        cluster_id_adj = {}  # 调整格式：{修改的目标：修改成的值}
        sd_node_adj = {}  # 调整格式：{修改的目标：变化量}
        sd_comm_adj = {}  # 调整格式：{修改的目标：变化量}
        sd_graph_adj = {'cut': 0, 'vol': 0}  # 调整格式：{修改的目标：变化量}

        # 1. 创建代理
        graph = deepcopy(self.graph)
        cluster_id = deepcopy(self.cluster_id)
        sd_comm = deepcopy(self.sd_comm)
        sd_graph = deepcopy(self.sd_graph)

        # 2. 初始化adjustment统计 & 统计涉及节点 & 统计新节点
        involved_nodes = set()
        for edge in incre_seq:

            new_nodes = set()
            # 统计涉及节点
            involved_nodes.add(edge[0])
            involved_nodes.add(edge[1])

            # 存储初始adjustment
            if edge[2] == '+':

                # 增量边的顶点都是已知顶点
                if edge[0] in graph.nodes and edge[1] in graph.nodes:

                    # 节点adjustment
                    d0 = graph.degree[edge[0]]
                    d1 = graph.degree[edge[1]]
                    # print(d0, d1)
                    incre_update(sd_node_adj, d0, -1)
                    incre_update(sd_node_adj, d1, -1)
                    incre_update(sd_node_adj, d0 + 1, 1)
                    incre_update(sd_node_adj, d1 + 1, 1)

                    # 社区adjustment
                    # print(edge)
                    c0 = cluster_id[edge[0]]
                    c1 = cluster_id[edge[1]]

                    if c0 != 'None' and c1 != 'None':  # 两个不是新节点
                        if c0 == c1:  # 社区内边
                            if c0 in sd_comm_adj:
                                sd_comm_adj[c0][1] += 2
                            else:
                                sd_comm_adj[c0] = [0, 2]

                            # 图adjustment
                            # sd_graph_adj['vol'] += 2

                        else:  # 社区间边
                            if c0 in sd_comm_adj:
                                sd_comm_adj[c0][0] += 1
                                sd_comm_adj[c0][1] += 1
                            else:
                                sd_comm_adj[c0] = [1, 1]

                            if c1 in sd_comm_adj:
                                sd_comm_adj[c1][0] += 1
                                sd_comm_adj[c1][1] += 1
                            else:
                                sd_comm_adj[c1] = [1, 1]

                            # # 图adjustment
                            # sd_graph_adj['cut'] += 2
                            # sd_graph_adj['vol'] += 2

                    elif c0 == 'None' or c1 == 'None':  # 边中至少有一个是incre seq中的新节点
                        if c0 == 'None' and c1 == 'None':  # 都是新节点
                            # 图adjustment
                            # sd_graph_adj['vol'] += 2
                            pass
                        else:  # 有一个是新节点
                            if c0 == 'None':
                                if c1 in sd_comm_adj:
                                    sd_comm_adj[c1][0] += 1
                                    sd_comm_adj[c1][1] += 1
                                else:
                                    sd_comm_adj[c1] = [1, 1]
                            elif c1 == 'None':
                                if c0 in sd_comm_adj:
                                    sd_comm_adj[c0][0] += 1
                                    sd_comm_adj[c0][1] += 1
                                else:
                                    sd_comm_adj[c0] = [1, 1]

                            # # 图adjustment
                            # sd_graph_adj['cut'] += 2
                            # sd_graph_adj['vol'] += 2

                # 增量边中有一个节点是未知的，图中不存在的
                elif edge[0] in graph.nodes or edge[1] in graph.nodes:
                    # 找到未知的节点（new）和已知的节点（end）
                    if edge[0] in graph.nodes:
                        new_node = edge[1]
                        end_node = edge[0]
                    else:
                        new_node = edge[0]
                        end_node = edge[1]
                    new_nodes.add(new_node)  # 将未知节点加入新节点集合

                    # 节点adjustment
                    d_end = graph.degree[end_node]
                    incre_update(sd_node_adj, d_end, -1)
                    incre_update(sd_node_adj, d_end + 1, 1)
                    incre_update(sd_node_adj, 1, 1)

                    # 社区adjustment，只统计已存在的一端的割边、社区变化量
                    c_end = cluster_id[end_node]
                    if c_end in sd_comm_adj:
                        sd_comm_adj[c_end][0] += 1
                        sd_comm_adj[c_end][1] += 1
                    else:
                        sd_comm_adj[c_end] = [1, 1]

                    # 节点-社区映射adjustment
                    if 'None' in id_clustering_adj:
                        id_clustering_adj['None'].append((new_node, '+'))
                    else:
                        id_clustering_adj['None'] = [(new_node, '+')]
                    cluster_id_adj[new_node] = 'None'

                    # 图adjustment
                    # sd_graph_adj['vol'] += 2

                # 加边的两个节点都是未知的
                else:
                    # 将两端的节点都加入新节点集合
                    new_nodes.add(edge[0])
                    new_nodes.add(edge[1])

                    # 节点adjustment
                    incre_update(sd_node_adj, 1, 1)
                    incre_update(sd_node_adj, 1, 1)

                    # 社区adjustment
                    pass

                    # 节点-社区映射adjustment
                    if 'None' in id_clustering_adj:
                        id_clustering_adj['None'].append((edge[0], '+'))
                        id_clustering_adj['None'].append((edge[1], '+'))
                    else:
                        id_clustering_adj['None'] = [(edge[0], '+')]
                        id_clustering_adj['None'] = [(edge[1], '+')]
                    cluster_id_adj[edge[0]] = 'None'
                    cluster_id_adj[edge[1]] = 'None'

                    # 图adjustment
                    # sd_graph_adj['vol'] += 2

                # 修改代理
                graph.add_edge(edge[0], edge[1])

            elif edge[2] == '-':
                # 假设减的边都是已存在的边

                # 节点adjustment
                d0 = graph.degree[edge[0]]
                d1 = graph.degree[edge[1]]
                incre_update(sd_node_adj, d0, -1)
                incre_update(sd_node_adj, d1, -1)
                incre_update(sd_node_adj, d0 - 1, 1)
                incre_update(sd_node_adj, d1 - 1, 1)

                # 社区adjustment
                c0 = cluster_id[edge[0]]
                c1 = cluster_id[edge[1]]
                if c0 == c1:  # 社区内边
                    if c0 in sd_comm_adj:
                        sd_comm_adj[c0][1] += -2
                    else:
                        sd_comm_adj[c0] = [0, -2]

                    # # 图adjustment
                    # sd_graph_adj['vol'] += -2

                else:  # 社区间边
                    if c0 in sd_comm_adj:
                        sd_comm_adj[c0][0] += -1
                        sd_comm_adj[c0][1] += -1
                    else:
                        sd_comm_adj[c0] = [-1, -1]

                    if c1 in sd_comm_adj:
                        sd_comm_adj[c1][0] += -1
                        sd_comm_adj[c1][1] += -1
                    else:
                        sd_comm_adj[c1] = [-1, -1]

                    # # 图adjustment
                    # sd_graph_adj['cut'] += -2
                    # sd_graph_adj['vol'] += -2

                # 修改代理
                graph.remove_edge(edge[0], edge[1])

            # 3. 更改代理映射，将新节点的社区设置为'None'
            # print(new_nodes)
            for node in new_nodes:
                cluster_id[node] = 'None'

        # 4. 开始节点偏移迭代
        # 生成次级代理映射
        cluster_id_pprox = deepcopy(cluster_id)
        while involved_nodes and iter_num > 0:
            # print(len(involved_nodes))
            # print(iter_num)
            new_involved_nodes = set()

            # 检查所有增量涉及到的节点
            # print()
            # print('Test: involved nodes', involved_nodes)

            # 代理在每次迭代后统一修改id，被修改节点的字典{node: new_id}
            new_node_id_iter = {}

            for node in involved_nodes:

                new_id = None  # 待偏移社区
                old_comm_id = cluster_id[node]  # 旧社区

                # -----------time evaluation-----------------------
                ns_starttime = time.time()

                # ------------------------------------------------------------------------------------
                # prefer_comm_ids = check_node(graph, node, cluster_id)  # 检查节点，获取节点偏好社区
                # # print('Test: node - prefer comm ids', node, prefer_comm_ids)
                #
                # # 决定偏移方向
                # # 偏好社区只有一个
                # if len(prefer_comm_ids) == 1:
                #     # 偏好社区不为原社区
                #     if prefer_comm_ids[0] != old_comm_id:
                #         # 待偏移社区定义为该唯一的偏好社区
                #         new_id = prefer_comm_ids[0]
                #     # 偏好社区为原社区
                #     else:
                #         # 不作任何改变
                #         pass
                #
                # # 偏好社区不止一个
                # elif len(prefer_comm_ids) > 1:
                #     # 检查结构熵变化量归于哪边较小，根据定理，倾向于将节点归于容量较小的社区
                #     # 注：假设增量很小，不会改变社区大小格局，因此一直使用原来的社区级别结构数据
                #     # 选择容量最小的偏好社区作为待偏移社区
                #
                #     # 过滤策略：如果偏好社区是原社区和一个其他社区，则还是归于原社区
                #     if old_comm_id in prefer_comm_ids:
                #         pass
                #     else:
                #         min_vol = 9999999
                #         for id in prefer_comm_ids:
                #             if sd_comm[id][1] < min_vol:
                #                 min_vol = sd_comm[id][1]
                #                 new_id = id
                # ------------------------------------------------------------------------------------

                # 偏好社区一定有且只有一个
                # 通过求解最优化问题来确定最优偏好社区
                # print('before', sd_comm_adj)
                prefer_comm_id = find_prefer_id(graph, node, cluster_id, sd_comm, sd_graph)  # 检查节点，获取节点偏好社区

                # if old_comm_id!= prefer_comm_id:
                #     print(f'Test: node {node} - comm id adjust {old_comm_id}->{prefer_comm_id}')

                # 如果偏好社区是原社区，则还是归于原社区（不发生偏移变化）
                if prefer_comm_id == old_comm_id:
                    new_id = None
                else:
                    new_id = prefer_comm_id

                # print('after',sd_comm_adj)
                # ------------------------------------------------------------------------------------

                ns_endtime = time.time()
                ns_time = ns_endtime - ns_starttime
                # if ns_time >= 0.0002:
                #     print(ns_time)

                # -----time evaluation-------------------------------------


                # 生成节点偏移调整：节点从旧社区id偏移到新社区id
                if new_id != None and old_comm_id != None:
                    # 存储id_clustering调整信息
                    if new_id in id_clustering_adj:
                        id_clustering_adj[new_id].append((node, '+'))
                    else:
                        id_clustering_adj[new_id] = [(node, '+')]

                    if old_comm_id in id_clustering_adj:
                        id_clustering_adj[old_comm_id].append((node, '-'))
                    else:
                        id_clustering_adj[old_comm_id] = [(node, '-')]

                    # 存储cluster_id调整信息
                    cluster_id_adj[node] = new_id

                    # 存储sd_comm调整信息
                    # 偏移的新旧社区都会有社区割边数量及容量的调整
                    # 首先获取偏移节点与各个社区的连边数量，这是基本数值
                    # 注意：为保证数据一致性，这些数值是根据本轮迭代已经发生的偏移实时更新的次级代理映射！
                    # 这种实时更新不会使得一个增量段变得内部-顺序敏感，效果只有保证一致性
                    # 内部顺序不敏感保证了同一时刻多个增量边的确定性，而这一确定性是给出准确结构熵值的关键
                    # 设l为旧社区连边数，n为新社区连边数，dt节点度数，这些都是更新后的值
                    l = 0
                    n = 0
                    dt = 0
                    # 根据次级代理映射确定社区连边数量
                    for neighbor in nx.neighbors(graph, node):
                        dt += 1
                        if cluster_id_pprox[neighbor] == old_comm_id and cluster_id_pprox[neighbor] != 'None':
                            l += 1
                        elif cluster_id_pprox[neighbor] == new_id and cluster_id_pprox[neighbor] != 'None':
                            n += 1

                    # 新节点从'None'偏移到社区
                    # 1. 新社区割边变化量：dt-2n
                    # 2. 新社区容量变化量：dt
                    # 3. 图割边变化量：2(dt-2n)
                    if old_comm_id == 'None' and new_id != 'None':
                        if verbose:
                            print(f'--Test (iter {3 - iter_num}): node {node} comm_id {old_comm_id}->{new_id}')
                            print('--Test: l n dt', l, n, dt)
                            print('--Test: new cut/vol delta', dt - 2 * n, dt)
                            print('--Test: original sd_comm_adj', sd_comm_adj)

                        if new_id in sd_comm_adj:
                            sd_comm_adj[new_id][0] += dt - 2 * n
                            sd_comm_adj[new_id][1] += dt
                        else:
                            sd_comm_adj[new_id] = [dt - 2 * n, dt]

                        if verbose:
                            print('--Test: updated sd_comm_adj', sd_comm_adj)

                            test_sd_comm = deepcopy(self.sd_comm)
                            for id in sd_comm_adj:
                                delta_cut, delta_vol = sd_comm_adj[id]
                                if id in test_sd_comm:
                                    test_sd_comm[id][0] += delta_cut
                                    test_sd_comm[id][1] += delta_vol
                                else:
                                    test_sd_comm[id] = [delta_cut, delta_vol]
                            print('--Test: current sd_comm', test_sd_comm)

                    # 节点在两个社区间偏移
                    # 1. 旧社区割边变化量：2l-dt
                    # 2. 新社区割边变化量：dt-2n
                    # 3. 旧社区容量变化量：-dt
                    # 4. 新社区容量变化量：dt
                    # 5. 图割边变化量：2l-2n
                    elif old_comm_id != 'None' and new_id != 'None':
                        if verbose:
                            print(f'--Test (iter {3 - iter_num}): node {node} comm_id {old_comm_id}->{new_id}')
                            print('--Test: l n dt', l, n, dt)
                            print('--Test: old cut/vol delta', 2 * l - dt, -dt)
                            print('--Test: new cut/vol delta', dt - 2 * n, dt)
                            print('--Test: original sd_comm_adj', sd_comm_adj)

                        if old_comm_id in sd_comm_adj:
                            sd_comm_adj[old_comm_id][0] += 2 * l - dt
                            sd_comm_adj[old_comm_id][1] += -dt
                        else:
                            sd_comm_adj[old_comm_id] = [2 * l - dt, -dt]

                        if new_id in sd_comm_adj:
                            sd_comm_adj[new_id][0] += dt - 2 * n
                            sd_comm_adj[new_id][1] += dt
                        else:
                            sd_comm_adj[new_id] = [dt - 2 * n, dt]

                        if verbose:
                            print('--Test: updated sd_comm_adj', sd_comm_adj)

                            test_sd_comm = deepcopy(self.sd_comm)
                            for id in sd_comm_adj:
                                delta_cut, delta_vol = sd_comm_adj[id]
                                if id in test_sd_comm:
                                    test_sd_comm[id][0] += delta_cut
                                    test_sd_comm[id][1] += delta_vol
                                else:
                                    test_sd_comm[id] = [delta_cut, delta_vol]
                            print('--Test: current sd_comm', test_sd_comm)

                    # 信息传递：检查所有邻居，如果邻居节点id不和该节点偏移后的社区一样，就会归入下一个迭代的涉及节点中
                    for neighbor in nx.neighbors(graph, node):
                        n_id = cluster_id[neighbor]
                        if n_id != new_id:
                            new_involved_nodes.add(neighbor)

                    new_node_id_iter[node] = new_id

                    # 实时修改次级代理映射
                    cluster_id_pprox[node] = new_id

            # 每次迭代后统一修改代理映射
            for node in new_node_id_iter:
                new_id = new_node_id_iter[node]
                cluster_id[node] = new_id

            involved_nodes = new_involved_nodes
            iter_num -= 1

        for id in sd_comm_adj:
            if id != 'None':
                sd_graph_adj['cut'] += sd_comm_adj[id][0]
                sd_graph_adj['vol'] += sd_comm_adj[id][1]

        return [sd_node_adj, sd_comm_adj, sd_graph_adj, id_clustering_adj, cluster_id_adj]

    # 生成一个朴素调整策略的adjustment
    def naive_adjust(self, incre_seq):
        id_clustering_adj = {}  # 调整格式：{社区id：[(变动的节点，‘+/-’),(),()]}
        cluster_id_adj = {}  # 调整格式：{修改的目标：修改成的值}
        sd_node_adj = {}  # 调整格式：{修改的目标：变化量}
        sd_comm_adj = {}  # 调整格式：{修改的目标：变化量}
        sd_graph_adj = {'cut': 0, 'vol': 0}  # 调整格式：{修改的目标：变化量}

        # 代理图
        graph = deepcopy(self.graph)
        graph = nx.Graph(graph)  # 由于存储的都是视图，所以要结冻
        cluster_id = deepcopy(self.cluster_id)

        for edge in incre_seq:
            if edge[2] == '+':

                # 加边
                if edge[0] in graph.nodes and edge[1] in graph.nodes:
                    c0 = cluster_id[edge[0]]
                    c1 = cluster_id[edge[1]]
                    d0 = graph.degree[edge[0]]
                    d1 = graph.degree[edge[1]]
                    incre_update(sd_node_adj, d0, -1)
                    incre_update(sd_node_adj, d1, -1)
                    incre_update(sd_node_adj, d0 + 1, 1)
                    incre_update(sd_node_adj, d1 + 1, 1)

                    if c0 == c1:  # 社区内边
                        if c0 in sd_comm_adj:
                            sd_comm_adj[c0][1] += 2
                        else:
                            sd_comm_adj[c0] = [0, 2]

                        sd_graph_adj['vol'] += 2

                    else:  # 社区间边
                        if c0 in sd_comm_adj:
                            sd_comm_adj[c0][0] += 1
                            sd_comm_adj[c0][1] += 1
                        else:
                            sd_comm_adj[c0] = [1, 1]

                        if c1 in sd_comm_adj:
                            sd_comm_adj[c1][0] += 1
                            sd_comm_adj[c1][1] += 1
                        else:
                            sd_comm_adj[c1] = [1, 1]

                        sd_graph_adj['cut'] += 2
                        sd_graph_adj['vol'] += 2

                    graph.add_edge(edge[0], edge[1])

                # 加节点
                elif edge[0] in graph.nodes or edge[1] in graph.nodes:
                    if edge[0] in graph.nodes:
                        new_node = edge[1]
                        end_node = edge[0]
                    else:
                        new_node = edge[0]
                        end_node = edge[1]
                    c_end = cluster_id[end_node]
                    d_end = graph.degree[end_node]
                    incre_update(sd_node_adj, d_end, -1)
                    incre_update(sd_node_adj, d_end + 1, 1)
                    incre_update(sd_node_adj, 1, 1)

                    if c_end in sd_comm_adj:
                        sd_comm_adj[c_end][1] += 2
                    else:
                        sd_comm_adj[c_end] = [0, 2]

                    if c_end in id_clustering_adj:
                        id_clustering_adj[c_end].append((new_node, '+'))
                    else:
                        id_clustering_adj[c_end] = [(new_node, '+')]
                    cluster_id_adj[new_node] = c_end

                    sd_graph_adj['vol'] += 2

                    graph.add_edge(edge[0], edge[1])
                    cluster_id[new_node] = c_end

                # 加孤立边：报错，朴素调整策略不支持临时的孤立边
                else:
                    assert 0


            else:

                # 减边：报错，朴素调整策略不支持减边
                assert 0

        return [sd_node_adj, sd_comm_adj, sd_graph_adj, id_clustering_adj, cluster_id_adj]

    # 用增量更新图数据（边，节点），策略无关
    def update_graph(self, incre_seq):
        for edge in incre_seq:
            # print(edge)
            if edge[2] == '+':
                self.graph.add_edge(edge[0], edge[1])
            elif edge[2] == '-':
                self.graph.remove_edge(edge[0], edge[1])

    # 基于adjustment动态修改存储的节点-社区映射
    def update_node_cluster(self, adjustment):
        id_clustering_adj = adjustment[3]
        cluster_id_adj = adjustment[4]
        # 更新节点-社区映射
        for id in id_clustering_adj:
            if id != 'None':
                for node in id_clustering_adj[id]:
                    if node[1] == '+':
                        self.id_clustering[id].append(node[0])
                    elif node[1] == '-':
                        self.id_clustering[id].remove(node[0])
        for node in cluster_id_adj:
            self.cluster_id[node] = cluster_id_adj[node]

    # 更新结构数据和结构表达式
    def update_sd_sexp(self, adjustment):
        sd_node_adj = adjustment[0]
        sd_comm_adj = adjustment[1]
        sd_graph_adj = adjustment[2]

        for d in sd_node_adj:
            delta_k = sd_node_adj[d]
            self.sexp_node += delta_k * d * math.log2(d)
            incre_update(self.sd_node, d, delta_k)

        for id in sd_comm_adj:
            if id != 'None':
                delta_cut, delta_vol = sd_comm_adj[id]
                cut, vol = self.sd_comm[id]
                self.sexp_comm -= (cut - vol) * math.log2(vol)
                # print('math', cut, delta_cut, vol, delta_vol)

                if vol + delta_vol != 0:
                    self.sexp_comm += (cut + delta_cut - vol - delta_vol) * math.log2(vol + delta_vol)

                if id in self.sd_comm:
                    self.sd_comm[id][0] += delta_cut
                    self.sd_comm[id][1] += delta_vol
                else:
                    self.sd_comm[id][0] = delta_cut
                    self.sd_comm[id][1] = delta_vol

        self.sexp_graph += - sd_graph_adj['cut']
        self.sd_graph['cut'] += sd_graph_adj['cut']
        self.sd_graph['vol'] += sd_graph_adj['vol']

    # 计算当前图的二维结构熵
    def calc_2dSE(self, verbose=False):
        SE = 0
        graph = self.graph
        m = len(graph.edges)
        total_g = 0
        for id in self.id_clustering:
            comm = self.id_clustering[id]
            if len(comm) != 0:
                g = get_cut(graph, comm)
                v = get_volume(graph, comm)
                SE += - g / (2 * m) * math.log2(v / (2 * m))
                for node in comm:
                    d = graph.degree[node]
                    SE += - d / (2 * m) * math.log2(d / v)
                    if verbose:
                        # print(f'true sd_node: node-{node} degree-{d}')
                        pass
                if verbose:
                    print(f'true sd_comm: comm-{id} cut-{g} vol-{v}')
                    total_g += g
        if verbose:
            print(f'true sd_graph: cut-{total_g} vol-{2 * m}')
            print()
        return SE

    # 绘制可视化图数据
    def show_graph(self):
        display_graph(self.graph)

    # 打印统计量
    def print_statistics(self):
        # print('sd_node:', self.sd_node)
        print('Init graph statistics:')
        print('comm num:', len(self.sd_comm))
        print('sd_graph:', self.sd_graph)
        return len(self.sd_comm)

    def get_comm_num(self):
        return len(self.sd_comm)

    def print_node_cluster(self):
        print('id-clustering', self.id_clustering)
        print('cluster-id', self.cluster_id)

    def print_adjustment(self, adjustment):
        # print('sd_node adj:', adjustment[0])
        print('sd_comm adj:', adjustment[1])
        print('sd_graph adj:', adjustment[2])
        # print('id-clustering adj:', adjustment[3])
        # print('cluster-id adj:', adjustment[4])
        pass


# 旧：node-shifting策略中的节点检查，用于检查涉及节点是否会偏移。
# 输入：节点id
# 输出：与该节点连边最多的社区id，不一定唯一
def check_node(graph, node, cluster_id):
    id_edge = {}

    for neighbor in nx.neighbors(graph, node):
        n_id = cluster_id[neighbor]
        if n_id != None and n_id != 'None':
            incre_update(id_edge, n_id, 1)
    # print(id_edge)

    max_keys = get_max_keys(id_edge)
    # print(max_keys)
    return max_keys


# node-shifting策略中确定最优偏移社区id的方法，求解最优化问题k* = argmin_k H_k。
# 输入：节点id
# 输出：最优偏移社区id
def find_prefer_id(graph, node, cluster_id, sd_comm, sd_graph):
    # starttime = time.time()
    old_id = cluster_id[node]
    dt = graph.degree[node]  # 目标节点度数
    m = sd_graph['vol']

    edge_k = {}  # 目标节点连接社区k的边数
    neighbor_ids = []

    # 求edge_k
    for neighbor in nx.neighbors(graph, node):
        n_id = cluster_id[neighbor]
        if n_id != None and n_id != 'None':
            incre_update(edge_k, n_id, 1)
            neighbor_ids.append(n_id)

    prefer_id = None
    min_value = 999999
    for id in neighbor_ids:
        g = sd_comm[id][0] + 0
        V = sd_comm[id][1] + 0

        # 获取当前社区对目标节点的连边数
        if id in edge_k:
            dtk = edge_k[id]
        else:
            dtk = 0

        # 如果当前社区是目标节点所在社区，则现需要移出该社区
        # print(g, V, dtk, dt)
        if id == old_id:
            g = g + dtk
            V = V - dt

        # 计算当前社区下，最优化问题右端表达式的值
        # print(g, V, dtk, dt)

        if V < 0:
            V = 0

        if V != 0:
            value = (g - V) * math.log2(V / (V + dt)) + 2 * dtk * math.log2((V + dtk) / (2 * m))
        else:
            value = 2 * dtk * math.log2((V + dtk) / (2 * m))

        # value1 = (g - V) * math.log2(V / (V + dt))
        # value2 = 2 * dtk * math.log2((V + dtk) / (2 * m))

        if value < min_value:
            prefer_id = id
            min_value = value
            # print(value1, value2, min_value, 'id', id)

    # verbose
    # if old_id != prefer_id:
    #     print(old_id, prefer_id, edge_k)

    # endtime = time.time()
    # print(round(endtime-starttime,6))
    # time.sleep(1)

    return prefer_id


def test1():
    cl = Clusteror()
    clustering = cl.infomap_cluster(example_g2)
    incre_graph = IncreGraph(example_g2, clustering)
    print('Original graph statistics:')
    incre_graph.print_statistics()
    print()

    incre_seq = seq_g2[0]
    naive_adjustment = incre_graph.naive_adjust(incre_seq)
    print('incre seq:', incre_seq)
    print('naive adjustment:', naive_adjustment)

    updated_SE = incre_graph.adjustment_based_SE_fast_computation(naive_adjustment, update=True)

    incre_graph.update_graph(incre_seq)
    incre_graph.update_node_cluster(naive_adjustment)
    se2d = incre_graph.calc_2dSE()

    print('New graph statistics:')
    incre_graph.print_statistics()
    print()
    print('adj based SE:', updated_SE)
    print('def based SE:', se2d)

    incre_graph.show_graph()


def test2():
    graph = demo_graph2
    print(graph.nodes)
    incre_seq = seq_dg2[0]
    cl = Clusteror()
    clustering = cl.infomap_cluster(graph)
    incre_graph = IncreGraph(graph, clustering)

    # 1. 打印原图结构数据和节点社区映射
    print('Original graph statistics & clusters:')
    incre_graph.print_statistics()
    incre_graph.print_node_cluster()
    # incre_graph.show_graph()
    print()

    # 2. 节点偏移策略adjustment生成 & 更新后结构熵的快速计算
    node_shifiting_adjustment = incre_graph.node_shifting_adjust(incre_seq, verbose=True)
    updated_SE = incre_graph.adjustment_based_SE_fast_computation(node_shifiting_adjustment, update=False)

    # 3. 打印生成的adjustment
    print('Node-shifting adjustment:')
    incre_graph.print_adjustment(node_shifiting_adjustment)
    print()

    # 4. 增量实际作用于原图
    incre_graph.update_graph(incre_seq)
    incre_graph.update_sd_sexp(node_shifiting_adjustment)
    incre_graph.update_node_cluster(node_shifiting_adjustment)

    # 5. 打印更新后图的结构数据和节点社区映射
    print('New graph statistics & clusters:')
    incre_graph.print_statistics()
    incre_graph.print_node_cluster()
    print()

    # 6. 定义法求更新后图的二维结构熵
    se2d = incre_graph.calc_2dSE(verbose=True)

    # 7. 打印两种方法的二维结构熵
    print('adj based SE:', updated_SE)
    print('def based SE:', se2d)


if __name__ == '__main__':
    test2()

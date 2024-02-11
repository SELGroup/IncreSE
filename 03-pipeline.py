# 实验pipeline，输出存储到output文件夹中

import pickle
import time
import networkx as nx
import numpy as np

from clusteror import Clusteror
from incre_graph import IncreGraph
from se_calculator import *
from copy import deepcopy
from utils.eval_tools import cluster_score_evaluation


# 动态-静态比较
def exp1(dataset_name, cluster_method_name):
    print(f'Experiment 1 on {dataset_name} and {cluster_method_name}')

    # 1. 选择并读取数据集
    init_graph, seq = load_dataset(dataset_name)
    print('init graph:', init_graph)
    print('incre seq number:', len(seq))

    # 2. 获取初始图的聚类结果，可选聚类方法，并构建incre_graph数据结构
    cl = Clusteror()
    cluster_method = getattr(cl, f'{cluster_method_name}_cluster')
    init_clustering = cluster_method(init_graph)
    incre_graph = IncreGraph(nx.Graph(init_graph), init_clustering)
    incre_graph.print_statistics()
    init_se = incre_graph.calc_2dSE()
    print('init 2dSE:', round(init_se, 4))
    print()

    # 3. 进行实时社区优化和实时结构熵评估（t+1以t为原始图进行更新）
    naive_graph = deepcopy(incre_graph)
    naive_results = []
    # naive_GI = []
    naive_times = []
    for i in range(len(seq)):
        # print(f'incre seq: {i + 1}, seq len: {len(seq[i])}')
        incre_seq = seq[i]
        naive_adj_start = time.time()
        naive_adjustment = naive_graph.naive_adjust(incre_seq)
        naive_adj_SE = naive_graph.adjustment_based_SE_fast_computation(naive_adjustment, update=False)
        naive_adj_end = time.time()
        naive_adj_time = naive_adj_end - naive_adj_start
        naive_results.append(naive_adj_SE)
        naive_times.append(naive_adj_time)
        naive_graph.update_sd_sexp(naive_adjustment)
        naive_graph.update_node_cluster(naive_adjustment)
        naive_graph.update_graph(incre_seq)
        print('naive SE:', round(naive_adj_SE, 4), 'naive time:', round(naive_adj_time, 2))
    print('finial node num:', len(naive_graph.graph.nodes), 'final edge num:', len(naive_graph.graph.edges))
    print()

    ns_graph = deepcopy(incre_graph)
    ns_results = []
    ns_times = []
    for i in range(len(seq)):
        print(f'incre seq: {i + 1}, seq len: {len(seq[i])}')
        incre_seq = seq[i]
        ns_adj_start = time.time()
        ns_adjustment = ns_graph.node_shifting_adjust(incre_seq, iter_num=5)
        ns_adj_SE = ns_graph.adjustment_based_SE_fast_computation(ns_adjustment, update=False)
        ns_adj_end = time.time()
        ns_adj_time = ns_adj_end - ns_adj_start
        ns_results.append(ns_adj_SE)
        ns_times.append(ns_adj_time)
        ns_graph.update_sd_sexp(ns_adjustment)
        ns_graph.update_node_cluster(ns_adjustment)
        ns_graph.update_graph(incre_seq)
        print('node-shifting SE:', round(ns_adj_SE, 4), 'node-shifting time:', round(ns_adj_time, 2))

    # 4. 进行静态社区优化和结构熵评估
    print()
    static_graph = deepcopy(incre_graph)
    static_results = []
    static_times = []
    static_comm = []
    for i in range(len(seq)):
        print(f'incre seq: {i + 1}, seq len: {len(seq[i])}')
        incre_seq = seq[i]
        static_start = time.time()
        static_graph.update_graph(incre_seq)
        static_clustering = cluster_method(static_graph.graph)
        static_SE = static_calc_2dSE(static_graph.graph, static_clustering)
        static_end = time.time()
        static_time = static_end - static_start
        static_results.append(static_SE)
        static_times.append(static_time)
        static_graph.get_init_statistics()
        static_comm.append(static_graph.get_comm_num())
        print('static SE:', round(static_SE, 4), 'def time:', round(static_time, 2), 'node num:', len(static_graph.graph.nodes))

    print(static_comm)
    print()
    with open(f'./output/exp1/{dataset_name}_{cluster_method_name}', 'wb') as save_file:
        save_content = {'dataset': dataset_name, 'static_method': cluster_method_name, 'time_num': len(seq)}
        save_content['naive_results'] = naive_results
        save_content['ns_results'] = ns_results
        save_content['static_results'] = static_results
        save_content['naive_times'] = naive_times
        save_content['ns_times'] = ns_times
        save_content['static_times'] = static_times
        save_content['static_comm'] = static_comm

        pickle.dump(save_content, save_file)


# 时间分析 & node-shifting超参数（迭代次数）选择对性能的影响/稳定性分析
def exp2(dataset_name, cluster_method_name):
    print(f'Experiment 2 on {dataset_name} and {cluster_method_name}')

    # 1. 选择并读取数据集
    init_graph, seq = load_dataset(dataset_name)
    print('init graph:', init_graph)
    print('incre seq number:', len(seq))

    # 2. 获取初始图的聚类结果，可选聚类方法，并构建incre_graph数据结构
    cl = Clusteror()
    cluster_method = getattr(cl, f'{cluster_method_name}_cluster')
    init_clustering = cluster_method(init_graph)
    incre_graph = IncreGraph(nx.Graph(init_graph), init_clustering)
    comm_num = incre_graph.print_statistics()
    init_se = incre_graph.calc_2dSE()
    print('init 2dSE:', round(init_se, 4))
    print()

    # 3. 进行实时社区优化和实时结构熵评估（t+1以t为原始图进行更新）
    ns_graph = deepcopy(incre_graph)
    ns_results = {3: [], 5: [], 7: [], 9: []}
    ns_times = {3: [], 5: [], 7: [], 9: []}
    for i in range(len(seq)):
        print(f'incre seq: {i + 1}, seq len: {len(seq[i])}')
        incre_seq = seq[i]
        ns_adjustment = None
        for iter_num in [3, 5, 7, 9]:
            ns_adjustment = None
            ns_adj_SE = None
            time_list = []
            for j in range(3):
                ns_adj_start = time.time()
                ns_adjustment = ns_graph.node_shifting_adjust(incre_seq, iter_num=iter_num)
                ns_adj_SE = ns_graph.adjustment_based_SE_fast_computation(ns_adjustment, update=False)
                ns_adj_end = time.time()
                ns_adj_time = ns_adj_end - ns_adj_start
                time_list.append(ns_adj_time)

            mean_time = np.array(time_list).mean()
            ns_results[iter_num].append(ns_adj_SE)
            ns_times[iter_num].append(mean_time)

        ns_graph.update_sd_sexp(ns_adjustment)
        ns_graph.update_node_cluster(ns_adjustment)
        ns_graph.update_graph(incre_seq)
        print('node-shifting mean SE:', round(ns_adj_SE, 4), 'node-shifting mean time (N=9):', round(mean_time, 2))

    with open(f'./output/exp2/{dataset_name}_{cluster_method_name}', 'wb') as save_file:
        save_content = {'dataset': dataset_name, 'static_method': cluster_method_name, 'time_num': len(seq)}
        save_content['ns_results'] = ns_results
        save_content['ns_times'] = ns_times
        save_content['comm_num'] = comm_num
        pickle.dump(save_content, save_file)


# 阈值分析: 5%~20%的边增量阈值开启动态调整-记录总时间消耗/最大结构熵波动
def exp3(dataset_name, cluster_method_name, threshold):
    print(f'Experiment 3 on {dataset_name} and {cluster_method_name}')

    # 1. 选择并读取数据集
    init_graph, seq = load_dataset(dataset_name)
    # print('init graph:', init_graph)
    # print('incre seq number:', len(seq))

    # 2. 获取初始图的聚类结果，可选聚类方法，并构建incre_graph数据结构
    cl = Clusteror()
    cluster_method = getattr(cl, f'{cluster_method_name}_cluster')
    init_clustering = cluster_method(init_graph)
    incre_graph = IncreGraph(nx.Graph(init_graph), init_clustering)
    incre_graph.print_statistics()
    init_se = incre_graph.calc_2dSE()
    # print('init 2dSE:', round(init_se, 4))
    # print()

    # 3. 进行带有更新阈值的实时社区优化和实时结构熵评估（t+1以t为原始图进行更新）
    # naive_graph = deepcopy(incre_graph)
    # naive_results = []
    # naive_times = []
    # naive_incre_seq = []
    # for i in range(len(seq)):
    #
    #     naive_incre_seq += seq[i]
    #     curnum_edge = naive_graph.sd_graph['vol']
    #     # print(f'time stamp: {i + 1}, seq_len: {len(seq[i])}, incre_percent: {round(100*len(naive_incre_seq)/curnum_edge, 2)}%')
    #     if len(naive_incre_seq)/curnum_edge >= threshold or i == 19:
    #         naive_adj_start = time.time()
    #         naive_adjustment = naive_graph.naive_adjust(naive_incre_seq)
    #         naive_adj_SE = naive_graph.adjustment_based_SE_fast_computation(naive_adjustment, update=False)
    #         naive_adj_end = time.time()
    #         naive_adj_time = naive_adj_end - naive_adj_start
    #         naive_results.append(naive_adj_SE)
    #         naive_times.append(naive_adj_time)
    #         # print('naive SE:', round(naive_adj_SE, 4), 'naive time:', round(naive_adj_time, 2))
    #
    #         naive_graph.update_sd_sexp(naive_adjustment)
    #         naive_graph.update_node_cluster(naive_adjustment)
    #         naive_graph.update_graph(naive_incre_seq)
    #         naive_incre_seq = []
    # print(f'Threshold: {threshold}, NA total time: {sum(naive_times)}, NA final SE: {naive_results[-1]}')

    ns_graph = deepcopy(incre_graph)
    ns_results = []
    ns_times = []
    ns_incre_seq = []
    for i in range(len(seq)):

        ns_incre_seq += seq[i]
        curnum_edge = ns_graph.sd_graph['vol']
        if len(ns_incre_seq) / curnum_edge >= threshold or i == 19:
            ns_adj_start = time.time()
            ns_adjustment = ns_graph.node_shifting_adjust(ns_incre_seq, iter_num=5)
            ns_adj_SE = ns_graph.adjustment_based_SE_fast_computation(ns_adjustment, update=False)
            ns_adj_end = time.time()
            ns_adj_time = ns_adj_end - ns_adj_start
            ns_results.append(ns_adj_SE)
            ns_times.append(ns_adj_time)
            # print('node-shifting SE:', round(ns_adj_SE, 4), 'node-shifting time:', round(ns_adj_time, 2))

            ns_graph.update_sd_sexp(ns_adjustment)
            ns_graph.update_node_cluster(ns_adjustment)
            ns_graph.update_graph(ns_incre_seq)
            ns_incre_seq = []
    print(f'Threshold: {threshold}, NS total time: {sum(ns_times)}, NS final SE: {ns_results[-1]}')

    # print()
    # with open(f'./output/exp1/{dataset_name}_{cluster_method_name}', 'wb') as save_file:
    #     save_content = {'dataset': dataset_name, 'static_method': cluster_method_name, 'time_num': len(seq)}
    #     save_content['naive_results'] = naive_results
    #     save_content['naive_times'] = naive_times
    #     save_content['ns_results'] = ns_results
    #     save_content['ns_times'] = ns_times
    #     pickle.dump(save_content, save_file)


# 噪声分析：分析Hawkes数据集初始状态的不同pac([0.02, 0.04, 0.06, 0.08, 0.1])情况下两种动态调整策略结构熵的变化曲线
# 改： 最终？/全部？ - 一个图五条线（5个pac），Infomap固定住，然后分别弄3个图表示NA、NS、TOA
# 最终：借用EXP1的输出模板
# def exp4():
#     method = 'infomap'
#     algo = ['NA', 'NS', 'TOA']
#     SE_list = {'NA': [], 'NS': [], 'TOA': []}
#     for i in range(3):
#
#         for j in [1, 2, 3, 4, 5]:
#             with open(f'datasets/noise_hawkes/init_state{j}', 'rb') as file1:
#                 init_graph = pickle.load(file1)
#             with open(f'datasets/noise_hawkes/hawkes{j}', 'rb') as file2:
#                 seq = pickle.load(file2)
#             incre_seq = []
#             for subseq in seq:
#                 incre_seq += subseq
#             incre_seq = incre_seq[:27454]
#             incre_size = int(len(incre_seq) / 10)
#
#
#             # 1. 选择并读取数据集
#             print('init graph:', init_graph)
#
#             # 2. 获取初始图的聚类结果，可选聚类方法，并构建incre_graph数据结构
#             cl = Clusteror()
#             cluster_method = getattr(cl, f'{method}_cluster')
#             init_clustering = cluster_method(init_graph)
#             incre_graph = IncreGraph(nx.Graph(init_graph), init_clustering)
#             incre_graph.print_statistics()
#             init_se = incre_graph.calc_2dSE()
#             print('init 2dSE:', round(init_se, 4))
#             print()
#
#             # 3. 进行实时社区优化和实时结构熵评估（t+1以t为原始图进行更新）
#             for k in range(10):
#                 subseq = incre_seq[incre_size * k: incre_size * (k + 1)]
#                 # print(len(subseq))
#
#                 SE_sublist = []
#                 if i == 0:
#                     naive_graph = deepcopy(incre_graph)
#                     naive_adjustment = naive_graph.naive_adjust(subseq)
#                     SE = naive_graph.adjustment_based_SE_fast_computation(naive_adjustment, update=False)
#
#                 if i == 1:
#                     ns_graph = deepcopy(incre_graph)
#                     ns_adjustment = ns_graph.node_shifting_adjust(subseq, iter_num=5)
#                     SE = ns_graph.adjustment_based_SE_fast_computation(ns_adjustment, update=False)
#
#                 # 4. 进行静态社区优化和结构熵评估
#                 if i == 2:
#                     static_graph = deepcopy(incre_graph)
#                     static_graph.update_graph(subseq)
#                     static_clustering = cluster_method(static_graph.graph)
#                     SE = static_calc_2dSE(static_graph.graph, static_clustering)
#
#                 SE_sublist.append(SE)
#
#             SE_list[algo[i]].append(SE)
#
#         SE_list = [NA_final_SE, NS_final_SE, TOA_final_SE]
        # with open(f'./output/exp4/{method}', 'wb') as save_file:
        #     pickle.dump(SE_list, save_file)


seq_num_dict = {'Cit-HepPh': 32, 'Cit-HepTh': 26, 'facebook': 29, 'flickr': 25, 'dblp_coauthorship': 120}
init_num_dict = {'Cit-HepPh': 12, 'Cit-HepTh': 6, 'facebook': 9, 'flickr': 15, 'dblp_coauthorship': 100}


def load_dataset(dataset_name):
    is_artificial = False
    is_noise = False
    if dataset_name in ['new_hawkes', 'sbm_hawkes', 'gaussian_hawkes']:
        is_artificial = True
    if dataset_name in ['hawkes1', 'hawkes2', 'hawkes3', 'hawkes4', 'hawkes5']:
        is_artificial = True
        is_noise = True

    seq = []  # 增量序列的序列
    if is_artificial:
        if is_noise:
            artificial_source = 'noise_hawkes'
            with open(f'datasets/{artificial_source}/init_state{dataset_name[-1]}', 'rb') as file1:
                init_graph = pickle.load(file1)
            with open(f'datasets/{artificial_source}/{dataset_name}', 'rb') as file2:
                seq = pickle.load(file2)
        else:
            artificial_source = dataset_name
            with open(f'datasets/{artificial_source}/init_state', 'rb') as file1:
                init_graph = pickle.load(file1)
            with open(f'datasets/{artificial_source}/hawkes', 'rb') as file2:
                seq = pickle.load(file2)
    else:
        init_graph_No = init_num_dict[dataset_name]
        with open(f'datasets/real_datasets/{dataset_name}/snapshots/graph{init_graph_No}', 'rb') as file1:
            init_graph = pickle.load(file1)
        incre_seq_num = seq_num_dict[dataset_name]
        for i in range(init_graph_No, incre_seq_num):
            with open(f'datasets/real_datasets/{dataset_name}/incre_seqs/ordered_incre_seq{i}', 'rb') as file2:
                seq.append(pickle.load(file2))

    return init_graph, seq


if __name__ == '__main__':
    # Exp 1&2&3 test
    # exp1('Cit-HepTh', 'infomap')
    # exp2('Cit-HepTh', 'infomap')
    # exp3('Cit-HepPh', 'infomap', 0.2)



    # Exp1 final process
    # ['hawkes'] #
    #
    dataset_list = ['sbm_hawkes', 'gaussian_hawkes'] # ['Cit-HepPh', 'dblp_coauthorship', 'facebook', 'hawkes']
    static_method_list = ['infomap', 'louvain', 'leiden']
    for dataset_name in dataset_list:
        for static_method in static_method_list:
            exp1(dataset_name, static_method)

    # Exp2 final process
    # dataset_list = ['Cit-HepPh']
    # static_method_list = ['louvain']
    # for dataset_name in dataset_list:
    #     for static_method in static_method_list:
    #         exp2(dataset_name, static_method)

    # Exp3 final process
    # for threshold in [0, 0.05, 0.1, 0.15, 0.2]:
    #     exp3('Cit-HepPh', 'infomap', threshold)

    # Exp4 final process
    # for i in [1,2,3,4,5]:
        # dataset_list = [f'hawkes{i}']
        # static_method_list = ['infomap']
        # for dataset_name in dataset_list:
        #     for static_method in static_method_list:
        #         exp1(dataset_name, static_method)

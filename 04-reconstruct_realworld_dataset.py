# 重构真实数据集--DynaMo文章中所提到的数据集，由文章作者所整理
# 重构格式：
# 对于每个数据集，有T个图快照（nx.Graph，存于snapshots文件夹，对每个时刻的图使用connectize函数，使得每张快照只保留最大的连通分量子图），
# 和T段增量序列（列表，存于incre_seqs文件夹，且有序化，即段内有序，每条加入的边都是连接在原有图上的），增量序列段间有序，段内逻辑上无序。
# 序列内格式：一个增量边为一三元组（端点1，端点2，+/-），+/-表示是增边还是减边
# 真实数据集增量序列个数CP：31，CT：25，facebook：28，DBLP：31，flickr：24，youtube：33

import os
import pickle
import time
import networkx as nx
import numpy as np
from copy import deepcopy

seq_num_dict = {'Cit-HepPh': 31, 'Cit-HepTh': 25, 'facebook': 28, 'youtube': 23, 'dblp_coauthorship': 120}
init_num_dict = {'Cit-HepPh': 11, 'Cit-HepTh': 5, 'facebook': 1, 'youtube': 13, 'dblp_coauthorship': 100}

def show_brief(dataset_name):
    print(dataset_name)
    path = f'./datasets/raw_data/{dataset_name}/ntwk'
    for i in range(len(os.listdir(path))):
        file_path = os.path.join(path, str(i+1))
        with open(file_path, 'r') as file:
            edges = file.readlines()
            print(i+1, len(edges))
            print(edges[:5])
    print()

def reconstruct_graph(dataset_name):
    path = f'./datasets/raw_data/{dataset_name}/ntwk'
    save_path = f'./datasets/real_datasets/{dataset_name}'
    for i in range(len(os.listdir(path))):
        file_path = os.path.join(path, str(i + 1))
        graph = nx.Graph()
        with open(file_path, 'r') as file:
            edges = file.readlines()
            for edge in edges:
                refined_edge = str(edge).split('\t')
                n1 = int(refined_edge[0])
                n2 = int(refined_edge[1].split('\n')[0])
                graph.add_edge(n1,n2)

            with open(save_path + f'/snapshots/graph{i+1}', 'wb') as graph_file:
                # 查找所有的连通分量
                connected_components = list(nx.connected_components(graph))
                # 获取最大的连通分量
                largest_connected_component = max(connected_components, key=len)
                # 获取这个最大的连通分量的子图
                largest_subgraph = graph.subgraph(largest_connected_component)
                # print(largest_subgraph)
                pickle.dump(largest_subgraph, graph_file)

        print(f'Graph {i+1} is generated. Raw Graph: {graph}, saved subgraph: {largest_subgraph}.')

def reconstruct_seq(dataset_name):

    path = f'./datasets/real_datasets/{dataset_name}/snapshots/graph'
    save_path = f'./datasets/real_datasets/{dataset_name}'

    # 数据集0时刻的图设为时刻x图的复制
    init_num = init_num_dict[dataset_name]
    with open(path + str(init_num), 'rb') as file:
        init_state = nx.Graph(pickle.load(file))

    # {0: g_0, 1: g_1, 2: g_2, ...}
    graph_dict = {init_num: init_state}

    for i in range(init_num, seq_num_dict[dataset_name]):
        file_path = path + str(i + 1)
        with open(file_path, 'rb') as file:
            graph_dict[i+1] = nx.Graph(pickle.load(file))

        # 存储t时刻的增量序列
        with open(save_path + f'/incre_seqs/ordered_incre_seq{i + 1}', 'wb') as seq_file:
            # 比较t时刻的图的构成边和t-1时刻的图的构成边，得到t时刻相对于t-1时刻的增量边
            incre_seq = []

            G_old = deepcopy(graph_dict[i])
            G_new = deepcopy(graph_dict[i+1])
            # print(G_old, G_new)

            for e in G_old.edges:
                if not G_new.has_edge(e[0], e[1]):
                    incre_seq.append((e[0], e[1], '-'))

            for e in G_new.edges:
                if not G_old.has_edge(e[0], e[1]):
                    incre_seq.append((e[0], e[1], '+'))

            incre_seq_len = len(incre_seq)
            saved_seq = []
            while incre_seq != []:
                for e in incre_seq:
                    if e[2] == '+':
                        if G_old.has_node(e[0]) or G_old.has_node(e[1]):
                            saved_seq.append(e)
                            incre_seq.remove(e)
                            G_old.add_edge(e[0], e[1])


            pickle.dump(saved_seq, seq_file)
            print(f'Incre_seq {i+1} is generated. Incre seq len: {incre_seq_len}, saved seq len: {len(saved_seq)}')

# Call reconstruct_graph and reconstruct_seq
def reconstruct_dataset(dataset_name):
    print(f'reconstruct_graph {dataset_name}')
    reconstruct_graph(dataset_name)
    print()
    print(f'reconstruct_seq {dataset_name}')
    reconstruct_seq(dataset_name)
    print()

# Show reconstructed dataset's information
def show_reconstructed_brief(dataset_name):
    print(dataset_name)
    path = f'./datasets/real_datasets/{dataset_name}/'


    # init graph
    init_num = init_num_dict[dataset_name]
    with open(path + f'snapshots/graph{init_num}', 'rb') as graph_file:
        graph = pickle.load(graph_file)
        print(graph)

    seq_lens = []
    for i in range(init_num, seq_num_dict[dataset_name]):
        with open(path + f'incre_seqs/ordered_incre_seq{i+1}', 'rb') as seq_file:
            seq = pickle.load(seq_file)
            l = len(seq)
            seq_lens.append(l)

    with open(path + f'snapshots/graph{seq_num_dict[dataset_name]}', 'rb') as graph_file:
        new_graph = pickle.load(graph_file)
        print(new_graph)

    print(f'mean incre nodes {(len(new_graph.nodes) - len(graph.nodes))/2}, mean incre edges {(len(new_graph.edges) - len(graph.edges))/2}')

# Special reconstruction of DBLP
def process_dblp():
    total_seq = []
    for i in range(15, 21):
        print(i)
        with open(f'./datasets/real_datasets/dblp_coauthorship/incre_seqs/ordered_incre_seq{i}', 'rb') as seq_file:
            seq = pickle.load(seq_file)
            total_seq += seq

    # Order total_seq
    with open('datasets/real_datasets/dblp_coauthorship/snapshots/graph100', 'rb') as file:
        init_graph = nx.Graph(pickle.load(file))

    nodes = deepcopy(set(init_graph.nodes))
    print(len(nodes))

    candidate_edges = []
    ordered_edges = []
    count = 0
    for edge in total_seq:
        if count%10000 == 0:
            print(count)

        if edge[0] in nodes or edge[1] in nodes:
            ordered_edges.append(edge)
            nodes.add(edge[0])
            nodes.add(edge[1])
        else:
            candidate_edges.append(edge)

        if candidate_edges != []:
            print(candidate_edges)

        for e in candidate_edges:
            if e[0] in nodes or e[1] in nodes:
                ordered_edges.append(e)
                nodes.add(e[0])
                nodes.add(e[1])

        count += 1

    print(len(nodes))

    ordered_total_seq = ordered_edges
    l = len(ordered_total_seq)
    for i in range(20):
        sub_l = int(l/20)
        if (i+1)*sub_l > l:
            subseq = ordered_total_seq[i * sub_l: l]
        else:
            subseq = ordered_total_seq[i*sub_l: (i+1)*sub_l]
        print(len(subseq))
        with open(f'./datasets/real_datasets/dblp_coauthorship/incre_seqs/ordered_incre_seq10{i}', 'wb') as save_file:
            pickle.dump(subseq, save_file)
        print(subseq[:5])
    print()

if __name__ == '__main__':
    dataset_list = ['Cit-HepPh', 'dblp_coauthorship', 'facebook']
    for dataset_name in dataset_list:
        show_reconstructed_brief(dataset_name)

    pass
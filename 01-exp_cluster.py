# 实验：节点聚类/社区发现
# 1. 根据数据集字符串读取数据集，构建encoding_tree数据结构（初始化的一维tree和原始的cor_graph），并准备好增量序列
# 2. 对encoding_tree进行结构熵极小化（例），得到社区划分和结构熵值
# 3. 检视一个增量序列，分析并增量式计算结构熵，顺势得到社区划分
# 4.1. 对图施加增量序列变为更新图
# 4.2. 对更新图重新进行结构熵极小化，得到社区划分和结构熵值

import time
from clusteror import Clusteror
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from se_calculator import *
from utils.example_graph import *
from utils.display_tools import *
import numpy as np
import pickle

# 加载处理好的数据集
def load_dataset(dataset):
    if dataset == 'hawkes' or dataset == 'random' or dataset == 'triad':
        with open('datasets/artifact_datasets/init_state', 'rb') as file1:
            init_graph = pickle.load(file1)
        with open(f'datasets/artifact_datasets/{dataset}', 'rb') as file2:
            seqs = pickle.load(file2)
    else:
        with open(f'datasets/real_datasets/init_state_{dataset}', 'rb') as file1:
            init_graph = pickle.load(file1)
        with open(f'datasets/real_datasets/{dataset}', 'rb') as file2:
            seqs = pickle.load(file2)
    return init_graph, seqs

# 将聚类结果转化为评分所需要的标签列表格式：一个列表，内容为每个节点所属社区的编号
def cluster2label(cluster_true, cluster_pred):
    node_num = 0
    for clust in cluster_true:
        node_num += len(clust)

    label_true = {}
    label_pred = {}

    for i in range(len(cluster_true)):
        for node in cluster_true[i]:
            label_true[node] = i

    for j in range(len(cluster_pred)):
        for node in cluster_pred[j]:
            label_pred[node] = j

    label_list_true = []
    label_list_pred = []

    for key in label_true:
        label_list_true.append(label_true[key])
        label_list_pred.append(label_pred[key])

    return label_list_true, label_list_pred

# 求两个聚类的最大分类准确率
def get_ACC(cluster_true, cluster_pred):
    node_num = 0
    for clust in cluster_true:
        node_num += len(clust)

    # 计算代价矩阵：t * p 的二维矩阵
    cost_matrix = np.zeros((len(cluster_true), len(cluster_pred)))
    for t_index in range(len(cluster_true)):
        for p_index in range(len(cluster_pred)):
            c1 = cluster_true[t_index]
            c2 = cluster_pred[p_index]
            cost_matrix[t_index][p_index] = len(set(c1).intersection(c2))

    # 计算行列匹配索引
    row_index, col_index = linear_sum_assignment(cost_matrix, maximize=True)

    # 计算最大得分
    ACC = 0
    for i in row_index:
        c1 = cluster_true[i]
        c2 = cluster_pred[col_index[i]]
        ACC += len(set(c1).intersection(c2))
    ACC = round(ACC / node_num, 2)

    return ACC

# 评估聚类效果，输入为两个聚类结果，每个聚类结果都是一个嵌套列表，子列表包含某一社区中的节点ID
def cluster_score_evaluation(cluster_true, cluster_pred):
    label_true, label_pred = cluster2label(cluster_true, cluster_pred)

    # ACC = metrics.accuracy_score
    ACC = get_ACC(cluster_true, cluster_pred)
    NMI = metrics.normalized_mutual_info_score(label_true, label_pred)
    ARI = metrics.adjusted_rand_score(label_true, label_pred)

    return ACC, ARI, NMI

# 单次聚类（对一个动态图的快照进行聚类），返回聚类结果
def single_cluster(graph, method):
    clusteror = Clusteror()
    cls_method = getattr(clusteror, f'{method}_cluster')
    start_time = time.time()
    clustering = cls_method(graph)
    end_time = time.time()
    time_cost = end_time - start_time
    return clustering

# 将t时刻的图施加t+1时刻的增量序列，变为t+1时刻的图并返回
def update_graph(graph: nx.Graph, single_seq: list):
    for edge in single_seq:
        if edge[2] == '+':
            graph.add_edge(edge[0],edge[1])
        elif edge[2] == '-':
            graph.remove_edge(edge[0], edge[1])

    return graph

# 静态聚类实验总过程：输入数据集名和聚类方法名，输出该数据集所有快照聚类所对应的时间消耗列表、结构熵列表以及三种评分列表
def exp_cluster(dataset, method):
    clustering_list = []  # 所有快照聚类结果(clustering)的列表
    time_list = []
    SE_list = []
    ACC_list = []
    ARI_list = []
    NMI_list = []

    # 读取数据集
    graph, seqs = load_dataset(dataset)

    # 开始静态评估
    print('----------------- eval process start -------------------')
    start_time = time.time()
    clustering = single_cluster(graph, method)  # 对原始图进行节点聚类
    SE = static_calc_2dSE(graph, clustering)  # 根据聚类结果计算原始图的二维结构熵
    end_time = time.time()
    time_cost = end_time - start_time
    print(round(time_cost, 2))

    # 记录1.时间，2.结构熵，3.聚类结果
    time_list.append(time_cost)
    SE_list.append(SE)
    clustering_list.append(clustering)

    for i in range(len(seqs)):
        start_time = time.time()
        graph = update_graph(graph, seqs[i])
        clustering = single_cluster(graph, method)
        SE = static_calc_2dSE(graph, clustering)
        end_time = time.time()
        time_cost = end_time - start_time
        print(round(time_cost, 2))

        time_list.append(time_cost)
        SE_list.append(SE)
        clustering_list.append(clustering)


    for clustering in clustering_list:
        ACC, ARI, NMI= cluster_score_evaluation(clustering)
        ACC_list.append(ACC)
        ARI_list.append(ARI)
        NMI_list.append(NMI)

    return time_list, ACC_list, ARI_list, NMI_list


if __name__ == '__main__':
    exp_cluster('hawkes', method='infomap')
    pass
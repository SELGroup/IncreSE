from sklearn import metrics
from scipy.optimize import linear_sum_assignment
import numpy as np

# 评估聚类效果，输入为两个聚类结果，每个聚类结果都是一个嵌套列表，子列表包含某一社区中的节点ID
def cluster_score_evaluation(cluster_true, cluster_pred):
    label_true, label_pred = cluster2label(cluster_true, cluster_pred)

    # ACC = metrics.accuracy_score
    ACC = get_ACC(cluster_true, cluster_pred)
    NMI = metrics.normalized_mutual_info_score(label_true, label_pred)
    ARI = metrics.adjusted_rand_score(label_true, label_pred)

    return ACC, ARI, NMI

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
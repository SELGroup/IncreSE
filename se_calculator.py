# 重构的se_graph，主要用于计算图的结构熵
from networkx.algorithms import cuts
import networkx as nx
import math


# 计算给定聚类结果的二维结构熵
def static_calc_2dSE(graph: nx.Graph, clustering):
    SE = 0
    m = len(graph.edges)
    for comm in clustering:
        g = get_cut(graph, comm)
        v = get_volume(graph, comm)
        SE += - g / (2 * m) * math.log2(v / (2 * m))
        for node in comm:
            d = graph.degree[node]
            SE += - d / (2 * m) * math.log2(d / v)
    return SE

def get_cut(graph, comm):
    return cuts.cut_size(graph, comm)

def get_volume(graph, comm):
    return cuts.volume(graph, comm)



# 集成多种图聚类方式的聚类器
# 包括：结构熵极小化、Louvain、Leiden、算法信息论算法、Infomap、CNM、GNN
# 输入：nx.Graph无向无权连通图
# 输出：聚类结果（嵌套列表）
import networkx as nx
from igraph import Graph
import networkx.algorithms.community as nx_comm
from encoding_tree import EncodingTree
from utils.example_graph import example_g1

class Clusteror:
    def __init__(self):

        pass

    def label_propagation_cluster(self, graph: nx.Graph):
        part = nx_comm.label_propagation_communities(graph)
        # print('partition num:', len(part))
        part = [list(x) for x in part]
        return part

    def louvain_cluster(self, graph: nx.Graph):
        part = nx_comm.louvain_communities(graph)
        # print('partition num:', len(part))
        part = [list(x) for x in part]
        return part

    def infomap_cluster(self, graph: nx.Graph):
        g = Graph.from_networkx(graph)
        part = Graph.community_infomap(g)
        result = [[g.vs['_nx_name'][j] for j in part[i]] for i in range(len(part))]
        return result

    # too fast
    def leading_eigenvector_cluster(self, graph: nx.Graph):
        g = Graph.from_networkx(graph)
        part = Graph.community_leading_eigenvector(g)
        result = [[g.vs['_nx_name'][j] for j in part[i]] for i in range(len(part))]
        return result

    # too fast
    def multilevel_cluster(self, graph: nx.Graph):
        g = Graph.from_networkx(graph)
        part = Graph.community_multilevel(g)
        result = [[g.vs['_nx_name'][j] for j in part[i]] for i in range(len(part))]
        return result

    # too slow
    def minSE_cluster(self, graph: nx.Graph):
        etc = EncodingTree(graph)
        etc.greedy_minSE_2d()
        return etc.get_2d_communities()

    # too slow
    def girvan_newman_cluster(self, graph: nx.Graph):
        part = nx_comm.girvan_newman(graph)
        # print('partition num:', len(part))
        part = [list(x) for x in part]
        return part

    # too slow
    def asyn_fluidc_cluster(self, graph: nx.Graph):
        part = nx_comm.asyn_fluidc(graph)
        # print('partition num:', len(part))
        part = [list(x) for x in part]
        return part

    # bad performance
    def leiden_cluster(self, graph: nx.Graph):
        g = Graph.from_networkx(graph)
        part = Graph.community_leiden(g, objective_function='modularity')
        result = [[g.vs['_nx_name'][j] for j in part[i]] for i in range(len(part))]
        return result

    # too fast & pending
    # def label_propagation_cluster(self, graph: nx.Graph):
    #     g = Graph.from_networkx(graph)
    #     part = Graph.community_label_propagation(g)
    #     result = [[g.vs['_nx_name'][j] for j in part[i]] for i in range(len(part))]
    #     return result


if __name__ == '__main__':
    cl = Clusteror()
    print(cl.infomap_cluster(example_g1))

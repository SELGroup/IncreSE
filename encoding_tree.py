import time
import networkx as nx
import math
from networkx.algorithms.community import louvain_communities
from utils.hierachy_pos import hierarchy_pos_beautiful as h_pos
from networkx.algorithms import cuts
import matplotlib.pyplot as plt

class EncodingTree:
    def __init__(self, graph: nx.Graph):
        self.cor_graph = graph  # 该编码树对应的图（引用，称为关联图）
        self.m = len(self.cor_graph.edges)  # 关联图的边数
        self.tree = nx.DiGraph()  # 编码树本体：一棵有向树，树的节点名称为'h高度-当前层高的节点序号'
        self.root = 'h0-1'  # 编码树根节点：一个字符串标识
        self.partition = []  # 嵌套列表表示的编码树
        self.SE = 0  # 编码树对应的结构熵


    # 遍历当前编码树所有节点，计算并存储编码树每个节点上的结构熵（作为节点属性），并记录总结构熵值
    def save_SE_every_node(self):
        total_SE = 0
        for node in self.tree.nodes:
            if node != 'h0-1':  # 当前节点不是根节点
                node_SE = self.calc_SE_node(node)
                self.tree.nodes[node]['entropy'] = node_SE
                total_SE += node_SE
            else:
                self.tree.nodes[node]['entropy'] = 'not exist'
        self.SE = total_SE

    # 计算当前编码树某一节点的结构熵
    def calc_SE_node(self, node):
        comm = self.tree.nodes[node]['community']
        parent = self.tree.predecessors(node).__next__()
        comm_parent = self.tree.nodes[parent]['community']
        g = self.get_cut(comm)
        v = self.get_volume(comm)
        v_parent = self.get_volume(comm_parent)
        node_SE = - g / (2 * self.m) * math.log2(v / (v_parent))
        return node_SE

    # 贪心结构熵极小化构建编码树(Li)(2d版本)
    def greedy_minSE_2d(self):
        # 初始化
        self.init_encoding_tree_2d()

        # 在h1层遍历所有节点对，选择使结构熵极小化的点对进行贪心融合；重复进行直到融合后无法减小结构熵
        self.merge_loop('h0-1')

    # 贪心结构熵极小化构建编码树(Li)(3d版本)：对每一个社区再进行一次二维的计算
    def greedy_minSE_3d(self):
        start_time = time.time()

        # 初始化
        self.init_encoding_tree_3d()

        # 在h1层遍历所有节点对，选择使结构熵极小化的点对进行贪心融合；重复进行直到融合后无法减小结构熵
        self.merge_loop('h0-1')
        
        # 在h2层对每棵子树遍历所有节点对，选择使结构熵极小化的点对进行贪心融合；重复进行直到融合后无法减小结构熵
        for subtree in list(self.tree.successors('h0-1')):
            self.merge_loop(subtree)

        end_time = time.time()

        # 打印构建时间
        # time_cost = end_time - start_time
        # node_num = len(self.cor_graph.nodes)
        # print(f'greedy minSE 3d time cost: {round(time_cost,2)}, graph node num: {node_num}, rate: {round(node_num/time_cost,2)}')

    # 循环使用融合算子
    def merge_loop(self, root):
        T = self.tree
        while True:
            h1_node_list = list(T.successors(root))
            delta_merge_SE = 0
            best_alpha = ''
            best_beta = ''
            for i in range(len(h1_node_list)):
                for j in range(i + 1, len(h1_node_list)):
                    # print(i,j)
                    alpha = h1_node_list[i]
                    beta = h1_node_list[j]
                    cur_dSE = self.merge_dSE(alpha, beta)
                    if cur_dSE < delta_merge_SE:
                        delta_merge_SE = cur_dSE
                        best_alpha = alpha
                        best_beta = beta

            if best_alpha != '' and best_beta != '':
                self.merge_operator(best_alpha, best_beta)
            else:
                break

    # 返回融合两个节点的结构熵变化量（不实际改变树结构，只计算融合后的结构熵差值）
    def merge_dSE(self, alpha, beta):
        alpha_comm = self.get_comm(alpha)
        beta_comm = self.get_comm(beta)
        # print('alpha:', alpha, 'comm:', alpha_comm)
        # print('beta:', beta, 'comm:', beta_comm)

        # 融合前的局部结构熵
        alpha_SE = self.calc_SE_node(alpha)
        beta_SE = self.calc_SE_node(beta)

        # print('alpha SE:', alpha_SE)
        # print('beta SE:', beta_SE)

        ori_child_SE = 0
        # alpha_childs = self.get_comm(alpha)
        # beta_childs = self.get_comm(beta)
        alpha_childs = list(self.tree.successors(alpha))
        beta_childs = list(self.tree.successors(beta))

        # print('alpha + beta:', alpha_childs + beta_childs)
        for child in alpha_childs + beta_childs:
            ori_child_SE += self.calc_SE_node(child)
        # print('ori child SE:', ori_child_SE)

        # 融合后alpha节点上的结构熵
        ab_comm = alpha_comm+beta_comm
        ab_volume = self.get_volume(ab_comm)
        ab_cut = self.get_cut(ab_comm)
        # print('ab_volume:', ab_volume, 'ab_cut:', ab_cut)

        ab_parent = list(self.tree.predecessors(alpha))[0]
        ab_parent_comm = self.tree.nodes[ab_parent]['community']
        ab_parent_volume = self.get_volume(ab_parent_comm)
        m = self.m

        # print('m:', m, 'ab_parent_volume:', ab_parent_volume)
        ab_SE = -ab_cut / (2 * m) * math.log2(ab_volume / ab_parent_volume)

        # 融合后的子节点结构熵
        new_child_SE = 0
        for child in alpha_comm + beta_comm:
            d = self.cor_graph.degree[child]
            new_child_SE += - d / (2 * m) * math.log2(d / ab_volume)

        return (ab_SE + new_child_SE) - (alpha_SE + beta_SE + ori_child_SE)

    # 融合算子：作用于兄弟节点，将两个节点合并为一个节点（将beta并入alpha）
    def merge_operator(self, alpha, beta):
        T = self.tree
        beta_childs = list(T.successors(beta))
        self.set_comm(alpha, self.get_comm(alpha) + self.get_comm(beta))
        for node in beta_childs:
            T.add_edge(alpha, node)
        T.remove_node(beta)

    # 返回联合两个节点的结构熵变化量（不实际改变树结构，只计算融合后的结构熵差值）
    # def combine_dSE(self, alpha, beta):
    #     return 0
    # 联合算子：作用于兄弟节点，在原父节点和它们之间增加一个共同的父节点
    # def combine_operator(self, alpha, beta):
    #     pass

    # 初始化一棵二层双节段的编码树，作为Li和Pan二维优化算法的起始状态
    def init_encoding_tree_2d(self):
        T = self.tree
        T.add_node('h0-1', community=list(self.cor_graph.nodes))
        count = 1
        for node in list(self.cor_graph.nodes):
            new_node1 = 'h1-' + str(count)
            T.add_node(new_node1, community=[node])
            T.add_edge('h0-1', new_node1)
            new_node2 = 'h2-' + str(count)
            T.add_node(new_node2, community=[node])
            T.add_edge(new_node1, new_node2)
            count += 1

    # 初始化一棵三层三节段的编码树，作为Li和Pan三维优化算法的起始状态
    def init_encoding_tree_3d(self):
        T = self.tree
        T.add_node('h0-1', community=list(self.cor_graph.nodes))
        count = 1
        for node in list(self.cor_graph.nodes):
            new_node1 = 'h1-' + str(count)
            T.add_node(new_node1, community=[node])
            T.add_edge('h0-1', new_node1)
            new_node2 = 'h2-' + str(count)
            T.add_node(new_node2, community=[node])
            T.add_edge(new_node1, new_node2)
            new_node3 = 'h3-' + str(count)
            T.add_node(new_node3, community=[node])
            T.add_edge(new_node2, new_node3)
            count += 1

    # Louvain社区发现算法构建二维编码树：赋予树节点名和community属性
    def louvain_minSE_2d(self):
        partition = louvain_communities(self.cor_graph)
        self.partition = partition
        T = self.tree
        T.add_node('h0-1', community=list(self.cor_graph.nodes))  # 根节点
        count = 1
        subcount = 1
        for comm in partition:
            new_node = 'h1-' + str(count)
            T.add_node(new_node, community=comm)
            T.add_edge('h0-1', new_node)

            for node in comm:
                subnew_node = 'h2-' + str(subcount)
                T.add_node(subnew_node, community=[node])
                T.add_edge(new_node, subnew_node)
                subcount += 1
            count += 1
        # print(T.nodes, T.edges)

    ####################################################################################
    ### 画图与输出 #######################################################################
    ####################################################################################

    # 绘制相关图G
    def show_graph(self):
        fig, ax = plt.subplots()
        nx.draw(self.cor_graph, ax=ax, with_labels=True)
        plt.show()

    # 保存图G的图片
    def save_graph(self, path = './figures/cor_graph.jpg'):
        fig, ax = plt.subplots()
        nx.draw(self.cor_graph, ax=ax, with_labels=False, node_size = 100)
        plt.savefig(path)

    # 绘制编码树
    def show_encoding_tree(self):
        fig, ax = plt.subplots()
        pos = h_pos(self.tree)
        nx.draw(self.tree, ax=ax, with_labels=True, pos=pos)
        plt.show()

    # 保存编码树T的图片
    def save_encoding_tree(self, path = './figures/encoding_tree.jpg'):
        fig, ax = plt.subplots()
        pos = h_pos(self.tree)
        nx.draw(self.tree, ax=ax, with_labels=False, pos=pos, node_size = 30)
        plt.savefig(path)

    # 打印相关图G的详细信息
    def print_graph(self):
        print(self.cor_graph.nodes)

    # 打印编码树的详细信息
    def print_encoding_tree(self):
        print(self.tree.nodes)
        for node in self.tree.nodes:
            print(node, self.tree.nodes[node]['community'])

    # 输出二维社区
    def get_2d_communities(self):
        communities = []
        for node in list(self.tree.successors('h0-1')):
            communities.append(self.get_comm(node))
        return communities

    ####################################################################################
    ### 辅助函数 #########################################################################
    ####################################################################################

    def get_cut(self, comm):
        return cuts.cut_size(self.cor_graph, comm)
    def get_volume(self, comm):
        return cuts.volume(self.cor_graph, comm)
    def get_comm(self, tree_node):
        return self.tree.nodes[tree_node]['community']
    def set_comm(self, tree_node, comm):
        self.tree.nodes[tree_node]['community'] = comm


if __name__ == '__main__':
    # example graph
    g = nx.Graph()
    g.add_edges_from([('a1', 'a2'), ('a1', 'a3'), ('a2', 'a3'),
                      ('a3', 'b1'), ('b1', 'b2'), ('b1', 'b4'),
                      ('b2', 'b3'), ('b3', 'b4'), ('b2', 'b4'),
                      ('b2', 'c3'), ('c1', 'c3'), ('c1', 'c2'), ('c2', 'c3')])
    nx.set_node_attributes(g, {'a1': 0, 'a2': 0, 'a3': 0, 'b1': 1, 'b2': 1, 'b3': 1, 'b4': 1, 'c1': 2, 'c2': 2, 'c3':2}, 'label')
    et = EncodingTree(g)
    et.greedy_minSE_2d()
    et.print_encoding_tree()
    print(et.get_2d_communities())
    # et.save_graph()
    # et.save_encoding_tree()
    # et.show_encoding_tree()
    # et.print_graph()


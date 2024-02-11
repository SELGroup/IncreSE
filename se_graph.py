# 结构熵图主函数，建立数据结构和增量算法
# 数据结构
# 增量序列、节点集、边集
import networkx as nx
from networkx.algorithms import cuts
import networkx.algorithms.community as nx_comm
import matplotlib.pyplot as plt
import math
import time


class SEGraph:
    def __init__(self, graph: nx.Graph):
        self.graph = graph.copy()
        self.division = {}  # 划分：{社区1：[节点集1]，社区2：[节点集2]，... }
        self.struc_data = {}  # 结构数据：{社区1：[容量1，割边1]，社区2：[容量2，割边2]，... }
        self.struc_data_expression = []  # 结构数据表达式：[表达式1，表达式2，表达式3]
        self.incre_data = []  # 增量统计数据：{size, cut_graph_incre, degree_incre, cut_incre, volume_incre}
        self.m = len(self.graph.edges)
        self.comm_No = 0
        self.update_division_Louvain()
        self.update_struc_data()
        self.calc_expression_SD()

    # 初始化社区划分（每个节点一个社区）
    def init_division(self):
        self.division = {}
        for node in self.graph.nodes:
            new_comm = node
            self.division[new_comm] = [node]
            self.graph.nodes[node]['comm'] = new_comm

    # 根据当前division重新统计计算结构数据
    def update_struc_data(self):
        self.struc_data = {}
        for vname in self.division.keys():
            comm = self.division[vname]
            volume = self.get_volume(comm)
            cut = self.get_cut(comm)
            self.struc_data[vname] = [volume, cut]

    # 使用Louvain社区发现算法更新社区划分，并更新节点社区属性
    def update_division_Louvain(self):
        part = nx_comm.louvain_communities(self.graph)
        dict = {}
        for node in part:
            comm = part[node]
            if comm in dict:
                dict[comm].append(node)
            else:
                dict[comm] = [node]
        self.division = dict
        for vname in self.division.keys():
            comm = self.division[vname]
            for node in comm:
                self.graph.nodes[node]['comm'] = vname

    # 根据存储的结构数据计算结构数据表达式
    def calc_expression_SD(self):
        sum_gv_log = 0
        sum_d_log_d = 0
        g_G = 0
        for v, g in self.struc_data.values():
            sum_gv_log += (g - v) * math.log2(v)
            g_G += g

        for node in self.graph.nodes:
            d = self.graph.degree[node]
            sum_d_log_d += d * math.log2(d)

        self.struc_data_expression = [sum_gv_log, sum_d_log_d, g_G]

    # 使用结构熵极小化贪心算法更新社区划分，并更新节点社区属性，由于树高为2，仅贪心执行融合算子
    def update_division_MinSE(self):
        m = self.m

        # 融合算子：接受两个社区名参数，返回融合后的结构熵差值（后-前），且不改变结构
        def Mg_operator(v1, v2):
            comm1 = self.division[v1]
            g1 = self.get_cut(comm1)
            v1 = self.get_volume(comm1)
            v1SE = - g1 / (2 * m) * math.log2(v1 / (2 * m))

            v1nodeSE = 0
            for node in comm1:
                d = self.graph.degree[node]
                v1nodeSE += -d / (2 * m) * math.log2(d / v1)

            comm2 = self.division[v2]
            g2 = self.get_cut(comm2)
            v2 = self.get_volume(comm2)
            v2SE = -g2 / (2 * m) * math.log2(v2 / (2 * m))

            v2nodeSE = 0
            for node in comm2:
                d = self.graph.degree[node]
                v2nodeSE += -d / (2 * m) * math.log2(d / v2)

            comm_merged = comm1 + comm2
            gm = self.get_cut(comm_merged)
            vm = self.get_volume(comm_merged)
            vmSE = -gm / (2 * m) * math.log2(vm / (2 * m))

            vmnodeSE = 0
            for node in comm_merged:
                d = self.graph.degree[node]
                vmnodeSE += -d / (2 * m) * math.log2(d / v2)

            delta_SE = vmSE + vmnodeSE - (v1SE + v1nodeSE + v2SE + v2nodeSE)
            return delta_SE

        self.init_division()
        while True:
            comm_num = len(self.division)
            delta_SE = 99999
            vm1 = None
            vm2 = None
            for i in range(comm_num):
                for j in range(i + 1, comm_num):
                    v1 = list(self.division.keys())[i]
                    v2 = list(self.division.keys())[j]
                    new_delta_SE = Mg_operator(v1, v2)
                    if new_delta_SE < delta_SE:
                        delta_SE = new_delta_SE
                        vm1 = v1
                        vm2 = v2

            if delta_SE < 0:
                # Do change the tree structure: Merge v1 & v2 -> v1
                for node in self.division[vm2]:
                    self.graph.nodes[node]['comm'] = vm1
                self.division[vm1] += self.division[vm2]
                self.division.pop(vm2)
            else:
                break

    # 联合算子：接受两个社区名参数，返回联合后的结构熵差值（后-前），且不改变结构
    # def Cb_operator(self, v1, v2):
    #     pass

    def show_division(self):
        print(self.division)

    def show_struc_data(self):
        print(self.struc_data)

    def show_incre_data(self):
        print(self.incre_data)

    def print_graph(self):
        fig, ax = plt.subplots()
        nx.draw(self.graph, ax=ax, with_labels=True)
        plt.show()

    # 计算一维结构熵
    def calc_1dSE(self):
        SE = 0
        self.m = len(self.graph.edges)
        for node in self.graph.nodes:
            d = self.graph.degree[node]
            SE += - d / (2 * self.m) * math.log2(d / (2 * self.m))
        return SE

    def get_cut(self, comm):
        return cuts.cut_size(self.graph, comm)

    def get_volume(self, comm):
        return cuts.volume(self.graph, comm)

    # 计算给定划分的二维结构熵
    def calc_2dSE(self):
        SE = 0
        self.m = len(self.graph.edges)
        for comm in self.division.values():
            g = self.get_cut(comm)
            v = self.get_volume(comm)
            SE += - g / (2 * self.m) * math.log2(v / (2 * self.m))
            for node in comm:
                d = self.graph.degree[node]
                SE += - d / (2 * self.m) * math.log2(d / v)
        return SE

    # 接受一个合法的增量序列，动态调整图和二维编码树（划分）以及结构数据
    def incre_seq_arrival(self, seq):
        # 遍历增量序列
        for edge in seq:
            # 图的动态调整
            self.graph.add_edge(edge[0], edge[1])

            # 插入新节点对二维编码树和结构数据的调整
            if 'comm' not in self.graph.nodes[edge[0]]:
                direct_comm = self.graph.nodes[edge[1]]['comm']
                self.graph.nodes[edge[0]]['comm'] = direct_comm
                self.division[direct_comm].append(edge[0])
                self.struc_data[direct_comm][0] += 2
            elif 'comm' not in self.graph.nodes[edge[1]]:
                direct_comm = self.graph.nodes[edge[0]]['comm']
                self.graph.nodes[edge[1]]['comm'] = direct_comm
                self.division[direct_comm].append(edge[1])
                self.struc_data[direct_comm][0] += 2
            else:
                # 插入新边对结构数据的调整
                if self.graph.nodes[edge[1]]['comm'] == self.graph.nodes[edge[0]]['comm']:
                    current_comm = self.graph.nodes[edge[1]]['comm']
                    self.struc_data[current_comm][0] += 2
                else:
                    comm0 = self.graph.nodes[edge[0]]['comm']
                    comm1 = self.graph.nodes[edge[1]]['comm']
                    self.struc_data[comm0][0] += 1
                    self.struc_data[comm1][0] += 1
                    self.struc_data[comm0][1] += 1
                    self.struc_data[comm1][1] += 1

    # 接受一个合法的增量序列，仅动态调整图
    def incre_seq_arrival_only_graph(self, seq):
        # 遍历增量序列
        for edge in seq:
            # 图的动态调整
            self.graph.add_edge(edge[0], edge[1])
        self.m = len(self.graph.edges)

    # 接受一个合法的增量序列，更新增量2d统计数据
    def get_incre_data(self, seq):
        # 增量统计数据
        size = len(seq)
        cut_graph_incre = 0
        degree_incre = {}  # 节点度变化量统计
        cut_incre = {}  # 社区割边变化量统计
        volume_incre = {}  # 社区容量变化量统计

        # 根据节点获取其所在社区
        temp_comm_dict = {}

        def get_comm(node):
            if node in self.graph.nodes:
                return self.graph.nodes[node]['comm']
            elif node in temp_comm_dict:
                return temp_comm_dict[node]
            else:
                return 'None'

        def incre_update(incre_dict, key, value):
            if key in incre_dict:
                incre_dict[key] += value
            else:
                incre_dict[key] = value

        # 遍历增量序列
        for edge in seq:
            n0 = edge[0]
            n1 = edge[1]

            comm0 = get_comm(n0)
            comm1 = get_comm(n1)

            # 两个节点不能同时不属于任何社区
            assert comm0 != 'None' or comm1 != 'None'

            # 节点度数变化
            incre_update(degree_incre, n0, 1)
            incre_update(degree_incre, n1, 1)

            # 在两个已知节点之间添加连边
            if comm0 != 'None' and comm1 != 'None':

                if comm0 == comm1:
                    incre_update(volume_incre, comm0, 2)
                else:
                    cut_graph_incre += 2
                    incre_update(volume_incre, comm0, 1)
                    incre_update(volume_incre, comm1, 1)
                    incre_update(cut_incre, comm0, 1)
                    incre_update(cut_incre, comm1, 1)

            elif comm0 == 'None' and comm1 != 'None':
                temp_comm_dict[n0] = comm1
                incre_update(volume_incre, comm1, 2)
            elif comm0 != 'None' and comm1 == 'None':
                temp_comm_dict[n1] = comm0
                incre_update(volume_incre, comm0, 2)

        self.incre_data = size, cut_graph_incre, degree_incre, cut_incre, volume_incre

    # 按照定义计算二维全局不变量 O(N)
    def calc_GI2d(self, n):
        GI = 0
        self.m = len(self.graph.edges)
        for comm in self.division.values():
            g = self.get_cut(comm)
            v = self.get_volume(comm)
            GI += - g / (2 * self.m + 2 * n) * math.log2(v / (2 * self.m + 2 * n))
            for node in comm:
                d = self.graph.degree[node]
                GI += - d / (2 * self.m + 2 * n) * math.log2(d / v)
        return GI



    # 根据存储的结构数据表达式计算二维全局不变量 O(1)
    def fast_GI2d(self, n):
        sum_gv_log, sum_d_log_d, g_G = self.struc_data_expression
        m = self.m
        GI = 1 / (2 * m + 2 * n) * (-sum_gv_log - sum_d_log_d + g_G * math.log2(2 * m + 2 * n))
        return GI

    # calculate LD1d
    def calc_LD1d(self):
        LD = 0
        size, cut_graph_incre, degree_incre, cut_incre, volume_incre = self.incre_data
        n = size
        m = self.m
        for node in degree_incre.keys():
            delta_d = degree_incre[node]
            if node in self.graph.nodes:
                d = self.graph.degree[node]
                LD += (d + delta_d) * math.log2(d + delta_d) - d * math.log2(d)
            else:
                LD += delta_d * math.log2(delta_d)

        LD += - 2 * n * math.log2(2 * m + 2 * n)

        # 乘系数 1/(2m+2n)
        LD *= - 1 / (2 * m + 2 * n)
        return LD

    # 按照增量分解公式和增量统计数据计算二维局部变化量
    def calc_LD2d(self):
        LD = 0
        size, cut_graph_incre, degree_incre, cut_incre, volume_incre = self.incre_data
        n = size
        m = self.m
        # part1
        part1 = cut_graph_incre * math.log2(2 * m + 2 * n)
        LD += part1
        # print('LD part1:', cut_graph_incre * math.log2(2 * m + 2 * n))

        # part2
        part2 = 0
        vname_list = list(set(list(cut_incre.keys()) + list(volume_incre.keys())))
        for vname in vname_list:
            g = self.struc_data[vname][1]
            v = self.struc_data[vname][0]
            if vname not in cut_incre:
                delta_g = 0
            else:
                delta_g = cut_incre[vname]
            if vname not in volume_incre:
                delta_v = 0
            else:
                delta_v = volume_incre[vname]
            part2 += (g - v) * math.log2(v) - (g - v + delta_g - delta_v) * math.log2(v + delta_v)
        # print('LD part2:', part2)
        LD += part2

        # part3
        part3 = 0
        for node in degree_incre.keys():
            delta_d = degree_incre[node]
            if node in self.graph.nodes:
                d = self.graph.degree[node]
                part3 += d * math.log2(d) - (d + delta_d) * math.log2(d + delta_d)
            else:
                part3 += - delta_d * math.log2(delta_d)
        # print('LD part3:', part3)
        LD += part3

        # 乘系数 1/(2m+2n)
        LD *= 1 / (2 * m + 2 * n)
        return LD

    # pipeline 1：接受一个合法的规模为T*n的增量序列，返回H2GI(n)、H2GI(2n)、...、H2GI(T*n)以及对应的LD1、LD2、...、LDn
    def pipeline_incre(self, seq, T):
        pipeline_start_time = time.time()
        GI_list = []
        LD_list = []
        SE_list = []
        self.update_division_Louvain()
        self.update_struc_data()
        self.calc_expression_SD()
        length = len(seq)
        num = int(length / T)
        for t in range(T):
            if (t + 1) * num > length:
                size = length
            else:
                size = (t + 1) * num
            subseq = seq[:size]
            self.get_incre_data(subseq)
            GI2d = self.fast_GI2d(size)
            LD2d = self.calc_LD2d()
            GI_list.append(GI2d)
            LD_list.append(LD2d)
            SE_list.append(GI2d + LD2d)
        plt.plot(range(0, T), LD_list)
        plt.show()
        plt.plot(range(0, T), SE_list)
        plt.show()

        pipeline_end_time = time.time()
        pipeline_time_cost = pipeline_end_time - pipeline_start_time
        print('pipeline 1 time cost: ' + str(pipeline_time_cost))

        return SE_list

    # pipeline 2：接受一个合法的规模为T*n的增量序列，分别返回T个阶段Louvain给出的社区划分及其对应的结构熵
    def pipeline_Louvain(self, seq, T):
        pipeline_start_time = time.time()
        SE_list = []
        self.update_division_Louvain()
        self.update_struc_data()
        length = len(seq)
        n = int(length / T)
        for t in range(T):
            if (t + 1) * n > length:
                size = length
            else:
                size = (t + 1) * n
            subseq = seq[:size]

            # 增量序列逐段到达，修改图编码树，并按定义计算结构熵
            self.incre_seq_arrival(subseq)
            SE_list.append(self.calc_2dSE())
        plt.plot(range(0, T), SE_list)
        plt.show()

        pipeline_end_time = time.time()
        pipeline_time_cost = pipeline_end_time - pipeline_start_time
        print('pipeline 2 time cost: ' + str(pipeline_time_cost))

        return SE_list

    # pipeline 3：接受一个合法的规模为T*n的增量序列，分别返回T个阶段结构熵极小化算法给出的社区划分及其对应的结构熵
    def pipeline_MinSE(self):
        pipeline_start_time = time.time()
        SE_list = []

        pipeline_end_time = time.time()
        pipeline_time_cost = pipeline_end_time - pipeline_start_time
        print('pipeline 3 time cost: ' + str(pipeline_time_cost))
        return SE_list


if __name__ == "__main__":
    # example graph
    g = nx.Graph()
    g.add_edges_from([('a1', 'a2'), ('a1', 'a3'), ('a2', 'a3'),
                      ('a3', 'b1'), ('b1', 'b2'), ('b1', 'b4'),
                      ('b2', 'b3'), ('b3', 'b4')])

    seg = SEGraph(g)
    seg.show_division()
    print("1dSE: ", seg.calc_1dSE())

    #############DEMO#############
    seg.division = {'v1': ['a1', 'a2', 'a3'], 'v2': ['b1', 'b2', 'b3', 'b4']}
    for node in seg.division['v1']:
        seg.graph.nodes[node]['comm'] = 'v1'
    for node in seg.division['v2']:
        seg.graph.nodes[node]['comm'] = 'v2'
    ##############################

    print("2dSE: ", seg.calc_2dSE())

    print('division (unchanged): ')
    seg.print_graph()
    seg.show_division()

    print('structure data (unchanged): ')
    seg.update_struc_data()
    seg.show_struc_data()

    incre_seq = [('a2', 'a5'), ('b2', 'b4'), ('a5', 'a3'), ('a5', 'b1')]
    # incre_seq = [('a1', 'b3'), ('a3', 'b4'), ('a1', 'a4')]
    print('incremental statistic data: size, cut_graph_incre, degree_incre, cut_incre, volume_incre')
    seg.get_incre_data(incre_seq)
    seg.show_incre_data()

    print('GI2d: ', seg.calc_GI2d(4))
    seg.calc_expression_SD()
    print('GI2d(fast): ', seg.fast_GI2d(4))
    print('LD2d: ', seg.calc_LD2d())

    print('incremental sequence arrived.')
    seg.incre_seq_arrival(incre_seq)

    print('division (changed): ')
    seg.print_graph()
    seg.show_division()

    # print('structure data (changed, updated): ')
    # # seg.update_struc_data()
    # seg.show_struc_data()

    print('SE2d: ', seg.calc_2dSE())

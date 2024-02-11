# [更新版]
# 制作数据集人工数据集（3个）
# 格式：对于每个数据集，有一个初始图（文件1，nx.Graph）和x段t时刻增量序列（文件2，列表），增量序列段间有序，段内逻辑上无序
# 序列内格式：一个增量边为一三元组（端点1，端点2，+/-），+/-表示是增边还是减边
import networkx as nx
import dataset_generator as dg
import igraph as ig
from utils.example_graph import example_g1, seq_g1
import numpy as np
import copy
import pickle

# Initial State Settings
init_num_list = [800, 1000, 1200, 1400, 1600] # node_num == 6000
init_pin = 0.05
init_pout = 0.001
init_seed = 12
init_comments = f'init_num_list = {init_num_list}, init_pin = {init_pin}, init_pout = {init_pout}, init_seed = {init_seed}\n'

# Graph Generating Processes Settings
incre_size = 238848 # 3x

# hawkes
hawkes_size = incre_size
hawkes_p = 0.95 # edge prob
hawkes_sample_num = 10

# --------------------------------- Artifact Dataset Generation ------------------------------------#
# generate random partition init state
def generate_random_partition_initial_state():
    generator = dg.dataset_generator()
    generator.make_random_partition_graph(init_num_list, init_pin, init_pout, seed=init_seed)
    with open('datasets/new_hawkes/init_state', 'wb') as file:
        pickle.dump(generator.graph, file)
    print('Random Partition Initial State Generation Finished')

# generate SBM (Stochastic Block Models) init state
def generate_SBM_initial_state():
    block_list = [800, 1000, 1200, 1400, 1600]
    edge_probs = np.random.uniform(0.001, 0.005, (5, 5))
    edge_probs = (edge_probs + edge_probs.T) / 2
    np.fill_diagonal(edge_probs, np.random.uniform(0.01, 0.05, 5))
    sbm_graph = nx.stochastic_block_model(block_list, edge_probs)
    with open('datasets/sbm_hawkes/init_state', 'wb') as file:
        pickle.dump(sbm_graph, file)
    print('SBM Initial State Generation Finished')

# generate gaussian random partition model init state
def generate_gaussian_initial_state():
    node_num = 6000
    mean_comm_size = 1000
    shape_parameter = 100
    p_in = 0.05
    p_out = 0.001
    gaussian_graph = nx.gaussian_random_partition_graph(node_num, mean_comm_size, shape_parameter, p_in, p_out)
    with open('datasets/gaussian_hawkes/init_state', 'wb') as file:
        pickle.dump(gaussian_graph, file)
    print('Gaussian Initial State Generation Finished')

# generate hawkes incremental sequence (cumulative sequence at time T)
def generate_hawkes_incre_seq(init_state = 'new'):
    with open(f'datasets/{init_state}_hawkes/init_state', 'rb') as file:
        init_graph = pickle.load(file)
    generator = dg.dataset_generator(init_graph)
    hawkes_generator = copy.deepcopy(generator)
    hawkes_seq = hawkes_generator.generate_seq_hawkes(hawkes_size, hawkes_p, hawkes_sample_num)
    np.save(f'datasets/{init_state}_hawkes/cumulative_seqs/hawkes_seq.npy', np.array(hawkes_seq))
    print('Hawkes Sequence Generation Finished')

# segment of cumulative sequence, also attach the '+/-' token to the end of each incremental edge
def seg_seq(seq_name, init_state = 'new'):
    seq = np.load(f'./datasets/{init_state}_hawkes/cumulative_seqs/{seq_name}_seq.npy').tolist()
    print(len(seq))
    for edge in seq:
        edge.append('+')
    new_seq = []
    single_length = int(len(seq)/20)
    for i in range(20):
        left = i * single_length
        right = (i+1) * single_length
        new_seq.append(seq[left: right])
    with open(f'datasets/{init_state}_hawkes/{seq_name}', 'wb') as file:
        pickle.dump(new_seq, file)

# -------------------- Generate Noise Hawkes （Independent）-----------------------------#
def generate_noise_init_state(count):
    num_list = [100, 100, 200, 200, 200]  # node_num == 800
    pin = 0.5 - 0.033*(count-1)
    pac = 0.01 * count
    seed = 12
    generator = dg.dataset_generator()
    generator.make_random_partition_graph(num_list, pin, pac, seed=seed)
    with open(f'datasets/noise_hawkes/init_state{count}', 'wb') as file:
        pickle.dump(generator.graph, file)
    print('Initial State Generation Finished')
    with open(f'datasets/noise_hawkes/init_state{count}', 'rb') as file1:
        init_graph = pickle.load(file1)
    print('total nodes:',len(init_graph.nodes),'total edges:', len(init_graph.edges))
    return len(init_graph.edges)

def generate_noise_hawkes(count):
    local_incre_size = 74714 # init edge: 37357~38175
    with open(f'datasets/noise_hawkes/init_state{count}', 'rb') as file:
        init_graph = pickle.load(file)
    generator = dg.dataset_generator(init_graph)
    hawkes_generator = copy.deepcopy(generator)
    hawkes_seq = hawkes_generator.generate_seq_hawkes(local_incre_size, hawkes_p, hawkes_sample_num)
    np.save(f'datasets/noise_hawkes/cumulative_seqs/hawkes_seq{count}.npy', np.array(hawkes_seq))
    print('Hawkes Sequence Generation Finished')

    seq = np.load(f'./datasets/noise_hawkes/cumulative_seqs/hawkes_seq{count}.npy').tolist()
    for edge in seq:
        edge.append('+')
    new_seq = []
    single_length = int(local_incre_size / 20)
    for i in range(20):
        left = i * single_length
        right = (i + 1) * single_length
        new_seq.append(seq[left: right])
    with open(f'datasets/noise_hawkes/hawkes{count}', 'wb') as file:
        pickle.dump(new_seq, file)

#------------------------------------Check Datasets---------------------------------------#
def check_init_state(init_state = 'new_hawkes'):
    with open(f'datasets/{init_state}/init_state', 'rb') as file1:
        init_graph = pickle.load(file1)
    print(init_graph)

def check_hawkes(init_state = 'new_hawkes'):
    with open(f'datasets/{init_state}/hawkes', 'rb') as file2:
        seq = pickle.load(file2)
    count = 0
    for subseq in seq:
        count += len(subseq)
    print(count)

if __name__ == '__main__':
    # Initial State Generation
    # generate_SBM_initial_state()
    # check_init_state('sbm_hawkes')
    # generate_gaussian_initial_state()
    # check_init_state('gaussian_hawkes')

    # SBM-Hawkes Incremental Sequence Generation & Segmentation
    generate_hawkes_incre_seq(init_state='sbm') # incre seq generation
    seg_seq('hawkes', init_state='sbm') # incre seq segmentation
    check_hawkes('sbm_hawkes')

    # Gaussian-Hawkes Incremental Sequence Generation & Segmentation
    generate_hawkes_incre_seq(init_state='gaussian')  # incre seq generation
    seg_seq('hawkes', init_state='gaussian')  # incre seq segmentation
    check_hawkes('gaussian_hawkes')





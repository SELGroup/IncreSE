# 制作数据集人工数据集（3个）
# 格式：对于每个数据集，有一个初始图（文件1，nx.Graph）和x段t时刻增量序列（文件2，列表），增量序列段间有序，段内逻辑上无序
# 序列内格式：一个增量边为一三元组（端点1，端点2，+/-），+/-表示是增边还是减边

import dataset_generator as dg
from utils.example_graph import example_g1, seq_g1
import numpy as np
import copy
import pickle

# # Initial State Settings - old
# init_num_list = [400, 600, 300, 200, 500] # node_num == 2000
# init_pin = 0.3
# init_pout = 0.01
# init_seed = 12
# init_comments = f'init_num_list = {init_num_list}, init_pin = {init_pin}, init_pout = {init_pout}, init_seed = {init_seed}\n'

# Initial State Settings - new
init_num_list = [300, 300, 300, 300, 300, 400, 400, 400, 400, 400, 500, 500, 500, 500, 500,
                 300, 300, 300, 300, 300, 400, 400, 400, 400, 400, 500, 500, 500, 500, 500,
                 300, 300, 300, 300, 300, 400, 400, 400, 400, 400, 500, 500, 500, 500, 500,
                 300, 300, 300, 300, 300, 400, 400, 400, 400, 400, 500, 500, 500, 500, 500] # node_num == 6000
init_pin = 0.008
init_pout = 0.0004
init_seed = 10
init_comments = f'init_num_list = {init_num_list}, init_pin = {init_pin}, init_pout = {init_pout}, init_seed = {init_seed}\n'

# Initial State Settings - noise
# init_num_list = [200, 300, 400, 500, 600] # node_num == 2000
# init_pin = 0.1
# init_pout = 0.02 # 0.02, 0.04, 0.06, 0.08, 0.1
# init_seed = 12
# init_comments = f'init_num_list = {init_num_list}, init_pin = {init_pin}, init_pout = {init_pout}, init_seed = {init_seed}\n'

# Graph Generating Processes Settings
incre_size = 238848 # 3x

# hawkes
hawkes_size = incre_size
hawkes_p = 0.95 # edge prob
hawkes_sample_num = 10

# triad
triad_size = incre_size
triad_p = 0.95 # edge prob

# pbp
random_size = incre_size
random_p = 0.9 # inner edge prob
random_q = 0.05 # outer edge prob

# ---------------------- Artifact Dataset Generation ----------------------#
# save_random_process init state
def generate_initial_state():
    generator = dg.dataset_generator()
    generator.make_random_partition_graph(init_num_list, init_pin, init_pout, seed=init_seed)
    with open('datasets/new_hawkes/init_state', 'wb') as file:
        pickle.dump(generator.graph, file)
    print('Initial State Generation Finished')

# save_random_process hawkes sequence (cumulative sequence at T)
def generate_hawkes_dataset():
    with open('datasets/new_hawkes/init_state', 'rb') as file:
        init_graph = pickle.load(file)
    generator = dg.dataset_generator(init_graph)
    hawkes_generator = copy.deepcopy(generator)
    hawkes_seq = hawkes_generator.generate_seq_hawkes(hawkes_size, hawkes_p, hawkes_sample_num)
    np.save('datasets/new_hawkes/cumulative_seqs/hawkes_seq.npy', np.array(hawkes_seq))
    # load: np.load('demo.npy').tolist()
    print('Hawkes Sequence Generation Finished')

# save_random_process triad clusure sequence
def generate_triad_dataset():
    with open('datasets/artifact_datasets/init_state', 'rb') as file:
        init_graph = pickle.load(file)
    generator = dg.dataset_generator(init_graph)
    triad_generator = copy.deepcopy(generator)
    triad_seq = triad_generator.generate_seq_triad(triad_size, triad_p)
    print(len(triad_seq))
    np.save('datasets/artifact_datasets/cumulative_seqs/triad_seq.npy', np.array(triad_seq))
    print('Triad Closure Sequence Generation Finished')

# save_random_process PBP sequence
def generate_pbp_dataset():
    with open('datasets/artifact_datasets/init_state', 'rb') as file:
        init_graph = pickle.load(file)
    generator = dg.dataset_generator(init_graph)
    random_generator = copy.deepcopy(generator)
    random_seq = random_generator.generate_seq_random(random_size, random_p, random_q)
    np.save('./datasets/artifact_datasets/random_seq_new.npy', np.array(random_seq))
    print('Random (PBP) Sequence Generation Finished')

# segment of the cumulative sequence (after the generation of the sequences), also attach the '+/-' token to the end of each incremental edge tuple
def seg_seq(seq_name):
    seq = np.load(f'./datasets/new_hawkes/cumulative_seqs/{seq_name}_seq.npy').tolist()
    print(len(seq))
    for edge in seq:
        edge.append('+')
    new_seq = []
    single_length = int(len(seq)/20)
    for i in range(20):
        left = i * single_length
        right = (i+1) * single_length
        new_seq.append(seq[left: right])
    with open(f'datasets/new_hawkes/{seq_name}', 'wb') as file:
        pickle.dump(new_seq, file)

# segment of the cumulative sequence (EXP)
def seg_seq_exp(seq_name):
    init_size = 100807
    reallen = incre_size
    seq = np.load(f'./datasets/new_hawkes/cumulative_seqs/{seq_name}_seq.npy').tolist()[:reallen]
    for edge in seq:
        edge.append('+')
    new_seq = []

    for i in range(20):
        left = int(init_size * pow(1.0717, i) - init_size)
        right = int(init_size * pow(1.0717, (i + 1)) - init_size)
        # print(left, right)
        new_seq.append(seq[left: right])

    with open(f'datasets/new_hawkes/{seq_name}', 'wb') as file:
        pickle.dump(new_seq, file)

# ---------------------- Test Dataset Creating ---------------------------#
def create_test_dataset():
    with open('datasets/test_datasets/init_state_test', 'wb') as file1:
        pickle.dump(example_g1, file1)
    with open('datasets/test_datasets/seq_test', 'wb') as file2:
        pickle.dump(seq_g1, file2)

# ---------------------- Dataset Printer ---------------------------------#
def print_test_dataset():
    with open('datasets/test_datasets/init_state_test', 'rb') as file1:
        init_graph = pickle.load(file1)
    with open('datasets/test_datasets/seq_test', 'rb') as file2:
        seq = pickle.load(file2)
    print(init_graph)
    print(seq)

def print_artifact_dataset(dataset):
    with open('datasets/artifact_datasets/init_state', 'rb') as file1:
        init_graph = pickle.load(file1)
    with open(f'datasets/artifact_datasets/{dataset}', 'rb') as file2:
        seq = pickle.load(file2)
    print(init_graph)
    print(len(seq))

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

def check_init_state(source):
    with open(f'datasets/{source}/init_state', 'rb') as file1:
        init_graph = pickle.load(file1)
    print(init_graph)

def check_hawkes(source):

    with open(f'datasets/{source}/hawkes', 'rb') as file2:
        seq = pickle.load(file2)
    count = 0
    for subseq in seq:
        count += len(subseq)
    print(count)

if __name__ == '__main__':

    # Noise Hawkes Generation
    # for i in [1,2,3,4,5]:
    #     generate_noise_hawkes(i)
    # check_hawkes('noise_hawkes')

    # New Hawkes Generation
    generate_initial_state() # initial state generation
    check_init_state('new_hawkes')

    # generate_hawkes_dataset() # incre seq generation
    # seg_seq('hawkes') # incre seq segmentation
    # seg_seq_exp('hawkes')
    # check_hawkes('new_hawkes')

    # seg_seq('hawkes')
    # create_test_dataset()
    # print_test_dataset()
    # print_artifact_dataset('random')

    pass




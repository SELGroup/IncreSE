# 绘制实验结果图像，接受output文件夹中的实验结果数据，输出实验图像到figures文件夹

import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 1. 绘制实时监控和社区优化应用的结构熵折线图
# 横轴：时间t
# 纵轴：结构熵
# 图线属性（三条线）：动态方法——初始静态解析方法（如Leiden）+NAGA+AIUA、初始静态解析方法+NSGA+AIUA；静态方法——初始静态解析方法
# 【额外图线】：NAGA的O(1)结构熵预测图线（全局不变量分析）
# 图属性（3*6）：不同静态解析方法*不同的数据集（3人工+3真实）
# 进阶：十次取均值方差，画箱型图/工字形折线图
def draw1():
    # 实验结果路径
    path = f'./output/exp1/'

    # 创建包含12个子图的图形，三行四列
    dataset_list = ['Cit-HepPh', 'dblp_coauthorship', 'facebook']
    static_method_list = ['infomap', 'louvain', 'leiden']
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(18, 9))
    fig.subplots_adjust(hspace=0.5)
    fig.subplots_adjust(wspace=0.15)
    for row in range(3):
        for col in range(3):
            with open(path + f'{dataset_list[row]}_{static_method_list[col]}', 'rb') as file:
                save_content = pickle.load(file)
                static_SE = np.array(save_content['static_results'])
                naive_SE = np.array(save_content['naive_results'])
                # print(f'{dataset_list[row]}_{static_method_list[col]}')
                # print(naive_SE[-1])
                ns_SE = np.array(save_content['ns_results'])
                # print(round(100*(ns_SE[-1]-naive_SE[-1])/naive_SE[-1], 2))
                axs[row, col].plot(range(1, 21), ns_SE, color='red', marker='o', label='NSGA+AIUA')
                axs[row, col].plot(range(1, 21), naive_SE, color = 'blue', marker= '^', label = 'NAGA+AIUA')
                axs[row, col].plot(range(1, 21), static_SE, color = 'gray', linestyle = '--', label = 'TOA')

                if dataset_list[row] == 'dblp_coauthorship':
                    axs[row, col].set_title(f'dataset = DBLP | static method = {static_method_list[col].capitalize()}')
                else:
                    axs[row, col].set_title(f'dataset = {dataset_list[row][0].capitalize() + dataset_list[row][1:]} | static method = {static_method_list[col].capitalize()}')

                axs[row, col].set_xlabel('Time Stamps', fontsize=12)
                if col == 0:
                    axs[row, col].set_ylabel('Structural Entropy', fontsize = 12)
                axs[row, col].set_xticks(range(1, 21))
                axs[row, col].legend()

    plt.savefig('./figures/exp1.pdf')
    # plt.savefig('./figures/exp1.jpg', dpi = 600)

# 1(EX). 绘制实时监控和社区优化应用的结构熵折线图（仅人工数据集）
# 纵向拉长并将一行的图绘制到一个图里面
def draw1_artificial():
    # 实验结果路径
    path = f'./output/exp1/'

    # 创建包含12个子图的图形，最后应是一行三列
    dataset_list = ['hawkes', 'sbm_hawkes', 'gaussian_hawkes']
    static_method_list = ['infomap', 'louvain', 'leiden']
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(18, 9))
    fig.subplots_adjust(hspace=0.5)
    fig.subplots_adjust(wspace=0.15)
    for row in range(3):
        for col in range(3):
            with open(path + f'{dataset_list[row]}_{static_method_list[col]}', 'rb') as file:
                save_content = pickle.load(file)
                static_SE = np.array(save_content['static_results'])
                naive_SE = np.array(save_content['naive_results'])
                # print(f'{dataset_list[row]}_{static_method_list[col]}')
                # print(naive_SE[-1])
                ns_SE = np.array(save_content['ns_results'])

                # offset_x = np.arange(20)
                # offset_y = 0.11 * offset_x
                # offset_ns = offset_y
                # offset_naive = offset_y
                # offset_static = offset_y

                # print(round(100*(ns_SE[-1]-naive_SE[-1])/naive_SE[-1], 2))
                axs[row, col].plot(range(1, 21), ns_SE-naive_SE, color='red', marker='o', label='NSGA+AIUA')
                axs[row, col].plot(range(1, 21), naive_SE-naive_SE, color='blue', marker='^', label='NAGA+AIUA')
                axs[row, col].plot(range(1, 21), static_SE-naive_SE, color='gray', linestyle = '--', marker = 'v', label='TOA')

                if dataset_list[row] == 'sbm_hawkes':
                    axs[row, col].set_title(f'dataset = SBM-Hawkes | static method = {static_method_list[col].capitalize()}')
                elif dataset_list[row] == 'gaussian_hawkes':
                    axs[row, col].set_title(f'dataset = Gaussian-Hawkes | static method = {static_method_list[col].capitalize()}')
                else:
                    axs[row, col].set_title(
                        f'dataset = {dataset_list[row][0].capitalize() + dataset_list[row][1:]} | static method = {static_method_list[col].capitalize()}')

                axs[row, col].set_xlabel('Time Stamps', fontsize=12)
                if col == 0:
                    axs[row, col].set_ylabel('SE - SE of NAGA+AIUA', fontsize=12)
                axs[row, col].set_xticks(range(1, 21))
                axs[row, col].legend()

    plt.savefig('./figures/exp1ex.pdf')
    # plt.savefig('./figures/exp1ex.jpg', dpi=600)
    pass

# 2. 绘制超参数研究实验图像-改成表格
# 横轴：时间t
# 纵轴：结构熵
# 图线属性（N条线）：初始静态解析方法+NSGA(迭代次数分别为1,2,...,N)+AIUA
# 图属性（暂定3*6）：不同静态解析方法*不同的数据集（3人工+3真实）
# 进阶：十次取均值方差，画箱型图/工字形折线图
def draw2_2():
    path = f'./output/exp2/'
    # 创建包含12个子图的图形，三行四列
    dataset_list = ['Cit-HepPh', 'dblp_coauthorship', 'facebook', 'hawkes']
    static_method_list = ['infomap', 'louvain', 'leiden']
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(24, 3))
    fig.subplots_adjust(wspace=0.18)
    for col in range(4):

        # ns_SE = {'infomap': {}, 'louvain': {}, 'leiden': {}}
        SE_display = {}
        time_display = {}
        for i in [3, 5, 7, 9]:
            SE_display[i] = []
            for static_method in static_method_list:
                with open(path + f'{dataset_list[col]}_{static_method}', 'rb') as file:
                    save_content = pickle.load(file)
                    SE_display[i].append(round(np.array(save_content['ns_results'][i]).mean(),4))
                    # print(f'{dataset_list[col]}_{static_method}: ', round(np.array(save_content['ns_results'][i]).mean(),4))

        print(dataset_list[col])
        print(SE_display)

        # axs[col].bar(x-0.3, SE_display[1], width=0.1, color='red', label='Iter = 1')
        # axs[col].bar(x - 0.15, SE_display[3], width=0.1, color='white', label='Iter = 3')
        # axs[col].bar(x, SE_display[5], width=0.1, color='blue', label='Iter = 5')
        # axs[col].bar(x + 0.15, SE_display[7], width=0.1, color='orange', label='Iter = 7')
        # axs[col].bar(x+0.3, SE_display[9], width=0.1, color='green', label='Iter = 9')
        #
        # axs[col].set_title(f'dataset = {dataset_list[col]}')
        # axs[col].set_xlabel('Static Method', fontsize=10)
        # axs[col].set_ylabel('Mean Structural Entropy', fontsize=10)
        # axs[col].legend()

    # plt.show()

    # fig.subplots_adjust(hspace=0.5)
    # fig.subplots_adjust(wspace=0.18)
    # for row in range(3):
    #     for col in range(4):
    #         with open(path + f'{dataset_list[col]}_{static_method_list[row]}', 'rb') as file:
    #             save_content = pickle.load(file)
    #             ns_SE = {}
    #             for i in [1,3,5,7,9]:
    #                 ns_SE[i] = np.array(save_content['ns_results'][i])
    #
    #             axs[row, col].plot(range(1, 21), ns_SE[1] - ns_SE[9], color='red', marker='o', label='Node-shifting')
    #             axs[row, col].plot(range(1, 21), ns_SE[5] - ns_SE[9], color='blue', marker='^', label='Node-shifting')
    #             axs[row, col].plot(range(1, 21), ns_SE[9] - ns_SE[9], color='green', marker='*', label='Node-shifting')
    #             axs[row, col].set_title(f'dataset = {dataset_list[col]} | static method = {static_method_list[row]}')
    #             axs[row, col].set_xlabel('Time Stamps', fontsize=12)
    #             axs[row, col].set_ylabel('Structural Entropy', fontsize=12)
    #             axs[row, col].set_xticks(range(1, 21))
    #             axs[row, col].legend()
    #
    # plt.show()

    # df = pd.DataFrame(
    #     columns=['dataset', 'static_method', 'dynamic_method', 'SE difference', 'Time cost', 'Time stamp'])
    # for file_name in os.listdir(path):
    #     print(file_name)
    #     with open(path + file_name, 'rb') as file:
    #         save_content = pickle.load(file)
    #         for i in range(3,save_content['time_num']):
    #             for iter_num in [1, 3, 5, 7]:
    #                 df = df.append({'dataset': save_content['dataset'], 'static_method': save_content['static_method'],
    #                                 'dynamic_method': f'node-shifting-{iter_num}',
    #                                 'SE difference': save_content['ns_results'][9][i] -
    #                                                  save_content['ns_results'][iter_num][i],
    #                                 'Time cost': save_content['ns_times'][iter_num][i], 'Time stamp': i},
    #                                ignore_index=True)
    # print(df)
    # df.to_excel('./output/exp2.xlsx')

# 2. 时间消耗评估图像绘制（柱状图）
# 横轴：
# 纵轴：时间消耗
# 图线属性（N条线）：
# 进阶：十次取均值方差，画箱型图/工字形折线图
def draw2():
    path = f'./output/exp2/'
    # 创建包含12个子图的图形，三行四列
    dataset_list = ['Cit-HepPh', 'dblp_coauthorship', 'facebook', 'hawkes']
    static_method_list = ['infomap', 'louvain', 'leiden']
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(13.7,3.6))
    fig.subplots_adjust(wspace=0.2)
    x = np.arange(3)
    for col in range(4):

        time_display = {}
        for i in [3, 5, 7, 9]:
            time_display[i] = []
            for static_method in static_method_list:
                with open(path + f'{dataset_list[col]}_{static_method}', 'rb') as file:
                    save_content = pickle.load(file)
                    time_display[i].append(np.array(save_content['ns_times'][i]).mean())

        naive_time_display = []
        static_time_display = []
        for static_method in static_method_list:
            with open(f'./output/exp1/{dataset_list[col]}_{static_method}', 'rb') as file:
                save_content = pickle.load(file)
                naive_time_display.append(np.array(save_content['ns_times']).mean())
                static_time_display.append(np.array(save_content['static_times']).mean())


        # axs[col].set_yscale('log')
        axs[col].bar(x - 0.3, naive_time_display, width=0.12, color='#5e8fb5', label='NAGA')
        axs[col].bar(x - 0.15, time_display[3], width=0.12, color='#fee08b', label='NSGA (Iter = 3)')
        axs[col].bar(x, time_display[5], width=0.12, color='#fdae61', label='NSGA (Iter = 5)')
        axs[col].bar(x + 0.15, time_display[7], width=0.12, color='#f46d43', label='NSGA (Iter = 7)')
        axs[col].bar(x + 0.3, time_display[9], width=0.12, color='#d73027', label='NSGA (Iter = 9)')
        # axs[col].bar(x + 0.25, static_time_display, width=0.08, color= 'darkgrey', label='TOA')


        axs[col].legend(fontsize = 9, loc = 'upper right')
        if dataset_list[col] != 'dblp_coauthorship':
            axs[col].set_title(dataset_list[col][0].capitalize() + dataset_list[col][1:])
        else:
            axs[col].set_title('DBLP')

        if col == 0:
            axs[col].set_ylabel('Time Cost (s)', fontsize=12)

        axs[col].set_xticks([0, 1, 2])
        axs[col].set_xticklabels(['Infomap', 'Louvain', 'Leiden'], fontsize = 12)
        # axs[col].set_xlabel('Static Method', fontsize=12)
        # axs[col].set_xticklabels([f'Infomap\n({round(static_time_display[0],2)}s)', f'Louvain\n({round(static_time_display[1],2)}s)', f'Leiden\n({round(static_time_display[2],2)}s)'], fontsize=12)

    axs[0].set_ylim([1, 1.5])
    axs[1].set_ylim([0.5, 0.85])
    axs[2].set_ylim([1.5, 2.3])
    axs[3].set_ylim([2.3, 3.6])
    fig.subplots_adjust(bottom=0.2)
    plt.savefig('./figures/exp2.pdf')
    # plt.savefig('./figures/exp2.jpg', dpi = 600)

    pass

def draw4():
    # 数据
    SE_list = {
        'NA': [],
        'NS': [],
        'TOA': []
    }

    static_method = 'infomap'
    algo = ['NA', 'NS', 'TOA']
    for i in [1,2,3,4,5]:
        with open(f'./output/exp4/hawkes{i}_{static_method}', 'rb') as file:
            save_content = pickle.load(file)
            naive_SE = np.array(save_content['naive_results'])
            SE_list['NA'].append(naive_SE)
            ns_SE = np.array(save_content['ns_results'])
            SE_list['NS'].append(ns_SE)
            static_SE = np.array(save_content['static_results'])
            SE_list['TOA'].append(static_SE)


    print(SE_list)

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 2.5))
    # fig.subplots_adjust(wspace=0.1)
    pac_values = [0.01, 0.02, 0.03, 0.04, 0.05]
    colors = ['r', 'b', 'g', 'y', 'grey']
    for col in range(3):
        for i in range(5):
            axs[col].plot(range(20), SE_list[algo[col]][i], color=colors[i], marker = 'o', ms = 2, label='$p_{ac}=$' + f'{pac_values[i]}')
            # axs[col].plot(pac_values, SE_list['NS'], color='r', marker = '^', label='NS')
            # axs[col].plot(pac_values, SE_list['TOA'], color='g', marker = 'v', label='TOA')

        if col == 0:
            axs[col].set_title('NAGA+AIUA', fontsize=10)
        if col == 1:
            axs[col].set_title('NSGA+AIUA', fontsize=10)
        if col == 2:
            axs[col].set_title('TOA', fontsize=10)

        axs[col].legend(fontsize=9)
        axs[col].set_xlabel('Time Stamps', fontsize=10)
        if col == 0:
            axs[col].set_ylabel('Structural Entropy', fontsize=10)


    # 显示图形
    plt.savefig('./figures/exp4.pdf', bbox_inches='tight')
    # plt.savefig('./figures/exp4.jpg', dpi=600, bbox_inches='tight')
    pass

def show_time():
    dataset_list = ['Cit-HepPh', 'dblp_coauthorship', 'facebook', 'hawkes']
    static_method_list = ['infomap', 'louvain', 'leiden']
    for dataset in dataset_list:
        for static_method in static_method_list:
            with open(f'./output/exp1/{dataset}_{static_method}', 'rb') as file:
                save_content = pickle.load(file)
                print(f'{dataset}_{static_method}')
                print('na_times', round(np.array(save_content['naive_times']).mean(),2))
                print('ns_times (N=5)', round(np.array(save_content['ns_times']).mean(),2))
                print('static_times', round(np.array(save_content['static_times']).mean(),2))
        print()



if __name__ == '__main__':
    draw1()
    draw1_artificial()

    # draw2_2()
    # draw2_2()
    # show_time()

    # import matplotlib.pyplot as plt
    # import numpy as np
    #
    # 数据
    # x = np.arange(5)
    # data1 = np.random.randint(1, 10, size=5)
    # data2 = np.random.randint(1, 10, size=5)
    # data3 = np.random.randint(1, 10, size=5)
    #
    # 绘制图形
    # fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    # for i in range(4):
    #     axs[i].bar(x - 0.3, data1, width=0.2, label='data1')
    #     axs[i].bar(x, data2, width=0.2, label='data2')
    #     axs[i].bar(x + 0.3, data3, width=0.2, label='data3')
    #     axs[i].set_xticks(x)
    #     axs[i].set_xticklabels(['A', 'B', 'C', 'D', 'E'])
    #     axs[i].legend()
    #
    # plt.show()
    # pass

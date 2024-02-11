from networkx.algorithms import cuts
from collections import Counter

def get_cut(graph, comm):
    return cuts.cut_size(graph, comm)

def get_volume(graph, comm):
    return cuts.volume(graph, comm)

# 字典中的值+=value，如果没有该key则新建一个key并赋值
def incre_update(incre_dict, key, value):
    if key in incre_dict:
        incre_dict[key] += value
    else:
        incre_dict[key] = value

# 取非负字典中值最大的几个重复元素的键，并返回
def get_max_keys(dict):
    max_value = 0
    results = []
    for key in dict:
        if dict[key] > max_value:
            results = [key]
            max_value = dict[key]
        elif dict[key] == max_value:
            results.append(key)
    # print(results)
    return results

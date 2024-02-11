import networkx as nx
import matplotlib.pyplot as plt

def display_graph(graph: nx.Graph):
    fig, ax = plt.subplots()
    nx.draw(graph, ax=ax, with_labels=True)
    plt.show()

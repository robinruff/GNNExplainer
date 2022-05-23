import os
import numpy as np
import pandas as pd
import networkx as nx

def load_mutagenicity(path, filter_invalid_molecules=True):

    Mutagenicity_A = os.path.join(path, "Mutagenicity_A.txt")
    Mutagenicity_NODE_LABELS = os.path.join(path, "Mutagenicity_node_labels.txt")
    Mutagenicity_EDGE_LABELS = os.path.join(path, "Mutagenicity_edge_labels.txt")
    Mutagenicity_GRAPH_INDICATOR = os.path.join(path, "Mutagenicity_graph_indicator.txt")
    Mutagenicity_GRAPH_LABELS = os.path.join(path, "Mutagenicity_graph_labels.txt")

    A = pd.read_csv(Mutagenicity_A, header=None)
    node_labels = pd.read_csv(Mutagenicity_NODE_LABELS, header=None)
    edge_labels = pd.read_csv(Mutagenicity_EDGE_LABELS, header=None)
    graph_indicator = pd.read_csv(Mutagenicity_GRAPH_INDICATOR, header=None)
    graph_labels = pd.read_csv(Mutagenicity_GRAPH_LABELS, header=None)

    all_graphs = nx.Graph()
    
    for i in range(len(node_labels)):
        one_hot_atom_embedding = [0] * 14 # There are 14 elements in the dataset
        one_hot_atom_embedding[node_labels[0][i]] = 1
        all_graphs.add_node(i+1, features=one_hot_atom_embedding)

    edges = pd.concat([A, edge_labels], axis=1)
    edges.columns = ['node1', 'node2', 'bond']
    for i, row in edges.iterrows():
        one_hot_bond_embedding = [0] * 3 # There are 3 bond types in the dataset
        one_hot_bond_embedding[row['bond']] = 1
        all_graphs.add_edge(row['node1'], row['node2'], features=one_hot_bond_embedding)

    graph_indicator['nodes'] = graph_indicator.index + 1
    graph_nodes = graph_indicator.groupby(by=0)['nodes'].apply(list).to_list()

    graphs = []
    for nodes in graph_nodes:
        graphs.append(all_graphs.subgraph(nodes).to_directed())
    
    graph_labels = np.array(graph_labels[0])
    graph_labels[graph_labels < 0] = 0
    labels = list(graph_labels)
    
    if filter_invalid_molecules:
        filtered_graphs = []
        filtered_labels = []
        for m, l in zip(graphs, labels):
            if nx.algorithms.components.number_connected_components(m.to_undirected()) == 1:
                filtered_graphs.append(m)
                filtered_labels.append(l)
        return filtered_graphs, filtered_labels
    
    return graphs, labels

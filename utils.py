from torch_geometric.datasets import Planetoid
import torch
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.sparse as sp
import os
from torch_geometric.utils import to_undirected
import networkx as nx


def load_data(dataset_name, seed, device, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, pre_splitted=False):

    if dataset_name in {'cora', 'citeseer', 'pubmed'}:

        dataset = Planetoid(root='data/'+dataset_name, name=dataset_name)
        data = dataset[0]
        num_features = dataset.num_features
        num_classes = dataset.num_classes
        if pre_splitted:
            idx_train, idx_val, idx_test = data.train_mask, data.val_mask, data.test_mask
            features = normalize_features(data.x).to(device)
        else:
            idx_train, idx_val, idx_test = split_nodes(data.y, train_ratio, val_ratio, test_ratio, seed)
            features = normalize_features(data.x).to(device)
        labels = data.y.to(device)

        adj = sp.coo_matrix((np.ones(data.edge_index.shape[1]), (data.edge_index[0, :], data.edge_index[1, :])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = adj.todense()
        adj = torch.tensor(adj).to(device)

        idx_train = idx_train.to(device)
        idx_val = idx_val.to(device)
        idx_test = idx_test.to(device)

    elif dataset_name in ['new_chameleon', 'new_squirrel']:
        edge = np.loadtxt('data/{}/edges.txt'.format(dataset_name), dtype=int)
        labels = np.loadtxt('data/{}/labels.txt'.format(dataset_name), dtype=int).tolist()
        features = np.loadtxt('data/{}/features.txt'.format(dataset_name))
        labels = torch.tensor(labels, dtype=torch.long)

        adj = sp.coo_matrix((np.ones(edge.shape[0]), (edge[:, 0], edge[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = adj.todense()
        adj = torch.tensor(adj).to(device)

        n = len(labels)
        idx = [i for i in range(n)]
        np.random.shuffle(idx)
        r0 = int(n * train_ratio)
        r1 = int(n * 0.6)
        r2 = int(n * 0.8)
        train = np.array(idx[:r0])
        val = np.array(idx[r1:r2])
        test = np.array(idx[r2:])

        features = normalize_features(features).to(device).to(torch.float32)
        num_features = features.shape[1]
        num_classes = 3
        labels = torch.LongTensor(labels).to(device)
        idx_train = torch.tensor(train, dtype=torch.int64).to(device)
        idx_val = torch.tensor(val, dtype=torch.int64).to(device)
        idx_test = torch.tensor(test, dtype=torch.int64).to(device)


    elif dataset_name in {'cornell', 'texas', 'wisconsin', 'chameleon', 'squirrel', 'film'}:
        graph_adjacency_list_file_path = os.path.join('data', dataset_name, 'out1_graph_edges.txt')
        graph_node_features_and_labels_file_path = os.path.join('data', dataset_name,
                                                                f'out1_node_feature_label.txt')
        edges_unordered = np.genfromtxt(graph_adjacency_list_file_path, dtype=int)
        edges = torch.tensor(edges_unordered, dtype=torch.int64).transpose(0, 1)
        edge_index = to_undirected(edges)

        G = nx.DiGraph()
        graph_node_features_dict = {}
        graph_labels_dict = {}

        if dataset_name == 'film':
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    feature_blank = np.zeros(932, dtype=np.uint16)
                    feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                    graph_node_features_dict[int(line[0])] = feature_blank
                    graph_labels_dict[int(line[0])] = int(line[2])
        else:

            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                    graph_labels_dict[int(line[0])] = int(line[2])

        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                if int(line[0]) not in G:
                    G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                               label=graph_labels_dict[int(line[0])])
                if int(line[1]) not in G:
                    G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                               label=graph_labels_dict[int(line[1])])
                G.add_edge(int(line[0]), int(line[1]))

        features = np.array([features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
        labels = np.array([label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])
        features = normalize_features(features).to(device)
        labels = torch.tensor(labels, dtype=torch.long).to(device)
        num_features = features.shape[1]
        num_classes = torch.unique(labels).size(0)


        adj = sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0, :], edge_index[1, :])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = adj.todense()
        adj = torch.tensor(adj).to(device)

        if pre_splitted:
            assert (pre_splitted in {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'})
            splits_file_path = 'splits/{}_split_0.6_0.2_{}.npz'.format(dataset_name, pre_splitted)
            print(splits_file_path)
            with np.load(splits_file_path) as splits_file:
                train = splits_file['train_mask']
                val = splits_file['val_mask']
                test = splits_file['test_mask']
                idx_train = torch.BoolTensor(train).to(device)
                idx_val = torch.BoolTensor(val).to(device)
                idx_test = torch.BoolTensor(test).to(device)
        else:
            num_nodes = features.shape[0]
            idx = [i for i in range(num_nodes)]
            np.random.shuffle(idx)
            r0 = int(num_nodes * train_ratio)
            r1 = int(num_nodes * (1-test_ratio))
            train = np.array(idx[:r0])
            val = np.array(idx[r0:r1])
            test = np.array(idx[r1:])

            idx_train = torch.tensor(train, dtype=torch.int64).to(device)
            idx_val = torch.tensor(val, dtype=torch.int64).to(device)
            idx_test = torch.tensor(test, dtype=torch.int64).to(device)
    return adj, features, labels, num_features, num_classes, idx_train, idx_val, idx_test

def split_nodes(labels, train_ratio, val_ratio, test_ratio, random_state):


    idx = torch.arange(labels.shape[0])
    idx_train, idx_test = train_test_split(idx, random_state=random_state, train_size=train_ratio+val_ratio, test_size=test_ratio)

    if val_ratio:
        idx_train, idx_val = train_test_split(idx_train, random_state=random_state, train_size=train_ratio/(train_ratio+val_ratio))
    else:
        idx_val = None

    return idx_train, idx_val, idx_test


def accuracy(logits, labels):

    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)

    return correct.item() * 1.0 / len(labels)

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return torch.tensor(mx)

def normalize_adjacency(mx):
    rowsum = mx.sum(1)
    r_inv = rowsum.pow(-1 / 2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv @ mx
    mx = mx @ r_mat_inv
    return mx


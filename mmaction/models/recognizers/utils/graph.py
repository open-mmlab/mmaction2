# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np


def get_hop_distance(num_node, edge, max_hop=1):
    adj_mat = np.zeros((num_node, num_node))
    for i, j in edge:
        adj_mat[i, j] = 1
        adj_mat[j, i] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [
        np.linalg.matrix_power(adj_mat, d) for d in range(max_hop + 1)
    ]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(adj_matrix):
    Dl = np.sum(adj_matrix, 0)
    num_nodes = adj_matrix.shape[0]
    Dn = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    norm_matrix = np.dot(adj_matrix, Dn)
    return norm_matrix


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


class Graph:
    """The Graph to model the skeletons extracted by the openpose.

    Args:
        layout (str): must be one of the following candidates
        - openpose: 18 or 25 joints. For more information, please refer to:
            https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        strategy (str): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition
        Strategies' in our paper (https://arxiv.org/abs/1801.07455).

        max_hop (int): the maximal distance between two connected nodes.
            Default: 1
        dilation (int): controls the spacing between the kernel points.
            Default: 1
    """

    def __init__(self,
                 layout='openpose-18',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        assert layout in [
            'openpose-18', 'openpose-25', 'ntu-rgb+d', 'ntu_edge', 'coco'
        ]
        assert strategy in ['uniform', 'distance', 'spatial', 'agcn']
        self.get_edge(layout)
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        """This method returns the edge pairs of the layout."""

        if layout == 'openpose-18':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5),
                             (13, 12), (12, 11), (10, 9), (9, 8), (11, 5),
                             (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0),
                             (17, 15), (16, 14)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'openpose-25':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (23, 22),
                             (22, 11), (24, 11), (11, 10), (10, 9), (9, 8),
                             (20, 19), (19, 14), (21, 14), (14, 13), (13, 12),
                             (12, 8), (8, 1), (5, 1), (2, 1), (0, 1), (15, 0),
                             (16, 0), (17, 15), (18, 16)]
            self.self_link = self_link
            self.neighbor_link = neighbor_link
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21),
                              (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                              (10, 9), (11, 10), (12, 11), (13, 1), (14, 13),
                              (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                              (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.self_link = self_link
            self.neighbor_link = neighbor_link
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        elif layout == 'ntu_edge':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                              (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                              (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                              (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                              (23, 24), (24, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2
        elif layout == 'coco':
            self.num_node = 17
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
                              [6, 12], [7, 13], [6, 7], [8, 6], [9, 7],
                              [10, 8], [11, 9], [2, 3], [2, 1], [3, 1], [4, 2],
                              [5, 3], [4, 6], [5, 7]]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 0
        else:
            raise ValueError(f'{layout} is not supported.')

    def get_adjacency(self, strategy):
        """This method returns the adjacency matrix according to strategy."""

        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[
                                    i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        elif strategy == 'agcn':
            A = []
            link_mat = edge2mat(self.self_link, self.num_node)
            In = normalize_digraph(edge2mat(self.neighbor_link, self.num_node))
            outward = [(j, i) for (i, j) in self.neighbor_link]
            Out = normalize_digraph(edge2mat(outward, self.num_node))
            A = np.stack((link_mat, In, Out))
            self.A = A
        else:
            raise ValueError('Do Not Exist This Strategy')

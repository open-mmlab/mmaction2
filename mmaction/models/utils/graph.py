# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import numpy as np
import torch


def k_adjacency(A: Union[torch.Tensor, np.ndarray],
                k: int,
                with_self: bool = False,
                self_factor: float = 1) -> np.ndarray:
    """Construct k-adjacency matrix.

    Args:
        A (torch.Tensor or np.ndarray): The adjacency matrix.
        k (int): The number of hops.
        with_self (bool): Whether to add self-loops to the
            k-adjacency matrix. The self-loops is critical
            for learning the relationships between the current
            joint and its k-hop neighbors. Defaults to False.
        self_factor (float): The scale factor to the added
            identity matrix. Defaults to 1.

    Returns:
        np.ndarray: The k-adjacency matrix.
    """
    # A is a 2D square array
    if isinstance(A, torch.Tensor):
        A = A.data.cpu().numpy()
    assert isinstance(A, np.ndarray)
    Iden = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return Iden
    Ak = np.minimum(np.linalg.matrix_power(A + Iden, k), 1) - np.minimum(
        np.linalg.matrix_power(A + Iden, k - 1), 1)
    if with_self:
        Ak += (self_factor * Iden)
    return Ak


def edge2mat(edges: List[Tuple[int, int]], num_node: int) -> np.ndarray:
    """Get adjacency matrix from edges.

    Args:
        edges (list[tuple[int, int]]): The edges of the graph.
        num_node (int): The number of nodes of the graph.

    Returns:
        np.ndarray: The adjacency matrix.
    """
    A = np.zeros((num_node, num_node))
    for i, j in edges:
        A[j, i] = 1
    return A


def normalize_digraph(A: np.ndarray, dim: int = 0) -> np.ndarray:
    """Normalize the digraph according to the given dimension.

    Args:
        A (np.ndarray): The adjacency matrix.
        dim (int): The dimension to perform normalization.
            Defaults to 0.

    Returns:
        np.ndarray: The normalized adjacency matrix.
    """
    # A is a 2D square array
    Dl = np.sum(A, dim)
    h, w = A.shape
    Dn = np.zeros((w, w))

    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)

    AD = np.dot(A, Dn)
    return AD


def get_hop_distance(num_node: int,
                     edges: List[Tuple[int, int]],
                     max_hop: int = 1) -> np.ndarray:
    """Get n-hop distance matrix by edges.

    Args:
        num_node (int): The number of nodes of the graph.
        edges (list[tuple[int, int]]): The edges of the graph.
        max_hop (int): The maximal distance between two connected nodes.
            Defaults to 1.

    Returns:
        np.ndarray: The n-hop distance matrix.
    """
    A = np.eye(num_node)

    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


class Graph:
    """The Graph to model the skeletons.

    Args:
        layout (str or dict): must be one of the following candidates:
            'openpose', 'nturgb+d', 'coco', or a dict with the following
            keys: 'num_node', 'inward', and 'center'.
            Defaults to ``'coco'``.
        mode (str): must be one of the following candidates:
            'stgcn_spatial', 'spatial'. Defaults to ``'spatial'``.
        max_hop (int): the maximal distance between two connected
            nodes. Defaults to 1.
    """

    def __init__(self,
                 layout: Union[str, dict] = 'coco',
                 mode: str = 'spatial',
                 max_hop: int = 1) -> None:

        self.max_hop = max_hop
        self.layout = layout
        self.mode = mode

        if isinstance(layout, dict):
            assert 'num_node' in layout
            assert 'inward' in layout
            assert 'center' in layout
        else:
            assert layout in ['openpose', 'nturgb+d', 'coco']

        self.set_layout(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.inward, max_hop)

        assert hasattr(self, mode), f'Do Not Exist This Mode: {mode}'
        self.A = getattr(self, mode)()

    def __str__(self):
        return self.A

    def set_layout(self, layout: str) -> None:
        """Initialize the layout of candidates."""

        if layout == 'openpose':
            self.num_node = 18
            self.inward = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                           (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                           (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
            self.center = 1
        elif layout == 'nturgb+d':
            self.num_node = 25
            neighbor_base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
                             (7, 6), (8, 7), (9, 21), (10, 9), (11, 10),
                             (12, 11), (13, 1), (14, 13), (15, 14), (16, 15),
                             (17, 1), (18, 17), (19, 18), (20, 19), (22, 8),
                             (23, 8), (24, 12), (25, 12)]
            self.inward = [(i - 1, j - 1) for (i, j) in neighbor_base]
            self.center = 21 - 1
        elif layout == 'coco':
            self.num_node = 17
            self.inward = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 5),
                           (12, 6), (9, 7), (7, 5), (10, 8), (8, 6), (5, 0),
                           (6, 0), (1, 0), (3, 1), (2, 0), (4, 2)]
            self.center = 0
        elif isinstance(layout, dict):
            self.num_node = layout['num_node']
            self.inward = layout['inward']
            self.center = layout['center']
        else:
            raise ValueError(f'Do Not Exist This Layout: {layout}')
        self.self_link = [(i, i) for i in range(self.num_node)]
        self.outward = [(j, i) for (i, j) in self.inward]
        self.neighbor = self.inward + self.outward

    def stgcn_spatial(self) -> np.ndarray:
        """ST-GCN spatial mode."""
        adj = np.zeros((self.num_node, self.num_node))
        adj[self.hop_dis <= self.max_hop] = 1
        normalize_adj = normalize_digraph(adj)
        hop_dis = self.hop_dis
        center = self.center

        A = []
        for hop in range(self.max_hop + 1):
            a_close = np.zeros((self.num_node, self.num_node))
            a_further = np.zeros((self.num_node, self.num_node))
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if hop_dis[j, i] == hop:
                        if hop_dis[j, center] >= hop_dis[i, center]:
                            a_close[j, i] = normalize_adj[j, i]
                        else:
                            a_further[j, i] = normalize_adj[j, i]
            A.append(a_close)
            if hop > 0:
                A.append(a_further)
        return np.stack(A)

    def spatial(self) -> np.ndarray:
        """Standard spatial mode."""
        Iden = edge2mat(self.self_link, self.num_node)
        In = normalize_digraph(edge2mat(self.inward, self.num_node))
        Out = normalize_digraph(edge2mat(self.outward, self.num_node))
        A = np.stack((Iden, In, Out))
        return A

    def binary_adj(self) -> np.ndarray:
        """Construct an adjacency matrix for an undirected graph."""
        A = edge2mat(self.neighbor, self.num_node)
        return A[None]

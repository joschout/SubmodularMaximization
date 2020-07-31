from typing import Set, TypeVar

from submodmax.abstract_optimizer import AbstractSubmodularFunction

import numpy as np

E = TypeVar('E')

VertexIndex = int


def calculate_cut_value(cut_from_set: Set[VertexIndex], graph: np.ndarray):

    cut_from_indices: np.ndarray = np.array(list(cut_from_set))
    n_vertices = graph.shape[0]
    cut_from_mask = np.zeros(n_vertices, dtype=bool)
    cut_from_mask[cut_from_indices] = True
    cut_to_mask = ~cut_from_mask

    cut_value = 0
    for vertex_index in cut_from_set:
        vertex_edge_row: np.ndarray = graph[vertex_index]
        vertex_edges_to = vertex_edge_row[cut_to_mask]
        cut_value += np.sum(vertex_edges_to)
    return cut_value


class MaxCutObjectiveFunction(AbstractSubmodularFunction):
    """
    A cut divides a graph into two disjunct set of vertices: the "from"-set and the "to"-set.
    The size of the cut is
        * for an unweighted graph: the number of edges spanning the cut
        * for a weighted graph: the sume of the weights of the edges spanning the cut.

    The Max-cut problem is defined as follows:
        Given a graph G, find a maximum cut.
    This problem is NP-hard.

    """

    def __init__(self, graph_adjacency_matrix: np.ndarray, value_for_empty_vertex_set: float):
        self.graph_adjacency_matrix: np.ndarray = graph_adjacency_matrix
        self.value_for_empty_vertex_set: float = value_for_empty_vertex_set

    def evaluate(self, input_set: Set[int]) -> float:
        if len(input_set) == 0:
            return self.value_for_empty_vertex_set
        else:
            return calculate_cut_value(cut_from_set=input_set, graph=self.graph_adjacency_matrix)

from typing import Set, TypeVar

from examples.max_cut.calculate_cut import calculate_cut_value
from submodmax.abstract_optimizer import AbstractSubmodularFunction

import numpy as np

E = TypeVar('E')


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

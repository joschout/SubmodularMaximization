from typing import List
from itertools import combinations, chain

import numpy as np

from examples.max_cut.max_cut_objective_function import calculate_cut_value

VIndex = int


def has_correct_shape(matrix: np.ndarray):
    if len(matrix.shape) != 2:
        return False
    if matrix.shape[0] != matrix.shape[1]:
        return False
    return True

# cut_from_set: Set[VIndex] = {0}


def powerset(iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    xs = list(iterable)
    # note we return an iterator rather than a list
    return chain.from_iterable(combinations(xs,n) for n in range(len(xs)+1))


def main():
    adjacency_matrix_lol: List[List[int]] = [[1, 1, 0],
                                             [1, 0, 1],
                                             [0, 1, 1]]

    adjacency_matrix_np = np.array(adjacency_matrix_lol)
    print(adjacency_matrix_np)

    for subset in powerset(range(len(adjacency_matrix_lol))):
        if len(subset) != 0:
            subset = set(subset)
            cut_value = calculate_cut_value(subset, adjacency_matrix_np)
            print(f"cut {subset} has cut value {cut_value}")


if __name__ == '__main__':
    main()





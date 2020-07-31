from typing import Set, Tuple, List

import numpy as np


from submodmax.randomized_double_greedy_search import RandomizedDoubleGreedySearch
from submodmax.abstract_optimizer import AbstractOptimizer

from examples.max_cut.calculate_cut import has_correct_shape
from examples.max_cut.max_cut_objective_function import MaxCutObjectiveFunction


def run_max_cut(adjacency_matrix: np.ndarray) -> Tuple[Set[int], float]:

    if not has_correct_shape(adjacency_matrix):
        raise Exception(f"Adjacency matrix does not have a correct shape: {adjacency_matrix.shape}")

    n_vertices = adjacency_matrix.shape[0]

    ground_set: Set[int] = set(range(n_vertices))
    submodular_objective_function = MaxCutObjectiveFunction(
        graph_adjacency_matrix=adjacency_matrix,
        value_for_empty_vertex_set=0
    )

    optimizer: AbstractOptimizer = RandomizedDoubleGreedySearch(
        objective_function=submodular_objective_function,
        ground_set=ground_set,
        debug=False
    )

    local_optimum: Set[int] = optimizer.optimize()
    value_local_optimum: float = submodular_objective_function.evaluate(local_optimum)

    return local_optimum, value_local_optimum


def main():
    adjacency_matrix_lol: List[List[int]] = [[1, 1, 0],
                                             [1, 0, 1],
                                             [0, 1, 1]]

    adjacency_matrix_np = np.array(adjacency_matrix_lol)
    local_optimum, value_local_optimum = run_max_cut(adjacency_matrix_np)

    true_optimum: Set[int] = {1}
    true_optimal_value: float = 2

    print(local_optimum)
    if true_optimum == local_optimum:
        print(f"Found correct local optimum with function value {value_local_optimum}: {local_optimum},")
    else:
        print(f"Found {local_optimum} with function value {value_local_optimum},"
              f" but should be {true_optimum} with function value {true_optimal_value}")


if __name__ == '__main__':
    main()
from typing import Set, TypeVar

from submodmax.abstract_optimizer import AbstractSubmodularFunction, AbstractOptimizer
from submodmax.randomized_double_greedy_search import RandomizedDoubleGreedySearch

E = TypeVar('E')


class AndreasKrauseExampleObjectiveFunction(AbstractSubmodularFunction):
    def evaluate(self, input_set: Set[int]) -> float:
        if input_set == set():
            return 0
        elif input_set == {1}:
            return -1
        elif input_set == {2}:
            return 2
        elif input_set == {1, 2}:
            return 0
        else:
            raise Exception(f"The input set was not expected: {input_set}")


def run_example():
    """
    The example from the tutorial slides at www.submodularity.org
    Originally implemented by Andreas Krause (krausea@gmail.com) in his SFO toolbox in Matlab.

    The function:
        | input_set | output |
        |-----------|--------|
        | {}        | 0      |
        | {1}       | -1     |
        | {2}       | 2      |
        | {1, 2}    | 0      |

    The ground set: { 1, 2 }
    """

    ground_set: Set[int] = {1, 2}
    submodular_objective_function = AndreasKrauseExampleObjectiveFunction()

    optimizer: AbstractOptimizer = RandomizedDoubleGreedySearch(
        objective_function=submodular_objective_function,
        ground_set=ground_set,
        debug=False
    )
    local_optimum: Set[int] = optimizer.optimize()
    true_optimum: Set[int] = {2}
    print(local_optimum)
    if true_optimum == local_optimum:
        print(f"Found correct local optimum: {local_optimum}")
    else:
        print(f"Found {local_optimum}, but should be {true_optimum}")


if __name__ == '__main__':
    run_example()

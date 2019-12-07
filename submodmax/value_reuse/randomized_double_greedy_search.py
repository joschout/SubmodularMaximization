from typing import Set, TypeVar

from submodmax.double_greedy_search_decision_strategy import RandomizedDoubleGreedySearchDecisionStrategy
from submodmax.value_reuse.abstract_double_greedy_search import AbstractDoubleGreedySearchValueReuse
from submodmax.value_reuse.abstract_optimizer import AbstractSubmodularFunctionValueReuse

E = TypeVar('E')


class RandomizedDoubleGreedySearch(AbstractDoubleGreedySearchValueReuse):
    """
    Randomized (Double Greedy) Unconstrained submodular maximization, by Buchbinder and Feldman

    The approximation ratio of this algorithm is 1/2.

    There are two options:
        either we add the current element to X_prev
        or we remove it from Y_prev
    A 'smooth' decision is taken greedily based on values a and b.

    IF both a and b are non-negative,
    it randomly chooses whether to include or exclude the current element,
        with a probability proportional to the ratio between a an b

    See also:
        Buchbinder, N., Feldman, M., Naor, J. S., & Schwartz, R. (2015).
        A tight linear time (1/2)-approximation for unconstrained submodular maximization.
        SIAM Journal on Computing, 44(5), 1384–1402. https://doi.org/10.1137/130929205

    """

    def __init__(self, objective_function: AbstractSubmodularFunctionValueReuse, ground_set: Set[E],
                 debug: bool = True):
        super().__init__(objective_function, ground_set, debug)

        self.class_name = "submodmax.RandomizedDoubleGreedySearch"
        self.decision_strategy = RandomizedDoubleGreedySearchDecisionStrategy()

    def should_update_X(self, a: float, b: float):
        return self.decision_strategy.should_update_X(a, b, self.debug)

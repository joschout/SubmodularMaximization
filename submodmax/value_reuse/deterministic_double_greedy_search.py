from typing import Set, TypeVar, Iterable

from submodmax.double_greedy_search_decision_strategy import DeterministicDoubleGreedySearchDecisionStrategy
from submodmax.value_reuse.abstract_double_greedy_search import AbstractDoubleGreedySearchValueReuse
from submodmax.value_reuse.abstract_optimizer import AbstractSubmodularFunctionValueReuse

E = TypeVar('E')


class DeterministicDoubleGreedySearch(AbstractDoubleGreedySearchValueReuse):
    """
    Deterministic (Double Greedy) Unconstrained submodular maximization, by Buchbinder and Feldman

    The approximation ration of this algorithm is 1/3.

    There are two options:
        either we add the current element to X_prev
        or we remove it from Y_prev
    The decision is done greedily based on the marginal gains/profits of each of the two options.

    See also:
        Buchbinder, N., Feldman, M., Naor, J. S., & Schwartz, R. (2015).
        A tight linear time (1/2)-approximation for unconstrained submodular maximization.
        SIAM Journal on Computing, 44(5), 1384–1402. https://doi.org/10.1137/130929205

    """

    def __init__(self, objective_function: AbstractSubmodularFunctionValueReuse, ground_set: Set[E],
                 debug: bool = True):
        super().__init__(objective_function, ground_set, debug)
        self.class_name = 'submodmax.value_reuse.DeterministicDoubleGreedySearch'
        self.decision_strategy = DeterministicDoubleGreedySearchDecisionStrategy()

    def should_update_X(self, a: float, b: float) -> bool:
        return self.decision_strategy.should_update_X(a, b, self.debug)

    def ground_set_iterator(self) -> Iterable[E]:
        return self.ground_set

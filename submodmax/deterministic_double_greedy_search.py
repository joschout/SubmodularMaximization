from typing import Set, TypeVar

from abstract_double_greedy_search import AbstractDoubleGreedySearch
from .abstract_optimizer import AbstractSubmodularFunction

E = TypeVar('E')


class DeterministicDoubleGreedySearch(AbstractDoubleGreedySearch):
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

    def __init__(self, objective_function: AbstractSubmodularFunction, ground_set: Set[E], debug: bool = True):
        super().__init__(objective_function, ground_set, debug)

        self.class_name = "submodmax.DeterministicDoubleGreedySearch"

    def should_update_X(self, a: float, b: float):

        should_update_X = a >=b

        if self.debug:
            print("\t\ta =", a)
            print("\t\tb =", b)

            if should_update_X:
                print("\ta >= b")
                print("\tUPDATE X_prev:")
            else:
                print("\ta < b")
                print("\tUPDATE Y_prev:")

        return should_update_X


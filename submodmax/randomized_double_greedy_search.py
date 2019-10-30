import warnings
from typing import Set, TypeVar

import numpy as np

from .abstract_double_greedy_search import AbstractDoubleGreedySearch
from .abstract_optimizer import AbstractSubmodularFunction

E = TypeVar('E')


class RandomizedDoubleGreedySearch(AbstractDoubleGreedySearch):
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

    def __init__(self, objective_function: AbstractSubmodularFunction, ground_set: Set[E], debug: bool = True):
        super().__init__(objective_function, ground_set, debug)

        self.class_name = "submodmax.RandomizedDoubleGreedySearch"

    def should_update_X(self, a: float, b: float):
        a_prime: float = max(a, 0)
        b_prime: float = max(b, 0)

        prob_boundary: float
        if a_prime == 0 and b_prime == 0:
            prob_boundary = 1
        else:
            prob_boundary = a_prime / (a_prime + b_prime)

        random_value: float = np.random.uniform()

        if self.debug:
            print("\t\ta =", a, "-> a' = ", a_prime)
            print("\t\tb =", b, "-> b' = ", b_prime)
            print("\t\tprob_bound =", prob_boundary)
            print("\t\trand_val   = ", random_value)

        should_update_X = random_value <= prob_boundary

        if self.debug:
            if should_update_X:
                print("\trandom_value <= prob_boundary")
                print("\tUPDATE X_prev:")
            else:
                print("\trandom_value > prob_boundary")
                print("\tUPDATE Y_prev:")

        return should_update_X

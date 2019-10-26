import warnings
from typing import Set, TypeVar

import numpy as np

from .abstract_optimizer import AbstractOptimizer, AbstractSubmodularFunction

E = TypeVar('E')


class RandomizedDoubleGreedySearch(AbstractOptimizer):
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

    def optimize(self) -> Set[E]:
        if self.debug:
            print("======================================================")
            print("START submodmax.RandomizedDoubleGreedySearch optimizer")
            print("======================================================")

        X_prev: Set[E] = set()
        Y_prev: Set[E] = self.ground_set.copy()

        f_on_X_prev: float = self.objective_function.evaluate(X_prev)
        f_on_Y_prev: float = self.objective_function.evaluate(Y_prev)

        ground_set_size: int = len(self.ground_set)

        if self.debug:
            print("initialization:")
            print("X0 : size: ", len(X_prev), "/", ground_set_size, ", f(S): ", f_on_X_prev)
            print("Y0:  size: ", len(Y_prev), "/", ground_set_size, ", f(S): ", f_on_Y_prev)

        elem: E
        for i, elem in enumerate(self.ground_set, 1):
            X_prev_plus_elem: Set[E] = X_prev | {elem}
            f_on_X_prev_plus_elem: float = self.objective_function.evaluate(X_prev_plus_elem)
            a: float = f_on_X_prev_plus_elem - f_on_X_prev

            Y_prev_minus_elem: Set[E] = Y_prev - {elem}
            f_on_Y_prev_minus_elem: float = self.objective_function.evaluate(Y_prev_minus_elem)
            b: float = f_on_Y_prev_minus_elem - f_on_Y_prev

            a_prime: float = max(a, 0)
            b_prime: float = max(b, 0)

            prob_boundary: float
            if a_prime == 0 and b_prime == 0:
                prob_boundary = 1
            else:
                prob_boundary = a_prime / (a_prime + b_prime)

            random_value: float = np.random.uniform()

            if self.debug:
                print()
                print("element ", i, "/", ground_set_size)
                print("\t X_prev   --> size: ", len(X_prev), ", f(S):", f_on_Y_prev)
                print("\t X" + str(i) + " + e" + str(i) + " --> size: ", len(X_prev_plus_elem), ", f(S):",
                      f_on_X_prev_plus_elem)

                print("\t Y_prev   --> size: ", len(Y_prev), ", f(S):", f_on_Y_prev)
                print("\t Y" + str(i) + " - e" + str(i) + " --> size: ", len(Y_prev_minus_elem), ", f(S):", f_on_Y_prev_minus_elem)

                print("\t\ta =", a, "-> a' = ", a_prime)
                print("\t\tb =", b, "-> b' = ", b_prime)
                print("\t\tprob_bound =", prob_boundary)
                print("\t\trand_val   = ", random_value)

            if random_value <= prob_boundary:
                X_prev = X_prev_plus_elem
                f_on_X_prev = f_on_X_prev_plus_elem
                # Y_prev stays the same
                if self.debug:
                    print("\trandom_value <= prob_boundary")
                    print("\tUPDATE X_prev:")
                    print("\tX_prev --> size:", len(X_prev), ", f(X_prev):", f_on_X_prev)
            else:
                # X_prev stays the same
                Y_prev = Y_prev_minus_elem
                f_on_Y_prev = f_on_Y_prev_minus_elem
                if self.debug:
                    print("\trandom_value > prob_boundary")
                    print("\tUPDATE Y_prev:")
                    print("\tY_prev --> size:", len(Y_prev), ", f(Y_prev):", f_on_Y_prev)

        warnings.warn("remove equality check")

        if not X_prev == Y_prev:
            raise Exception("both sets should be equal")

        if self.debug:
            print("-- finished iteration --")
            print("X_prev --> size:", len(X_prev), ", f(X_prev):", f_on_X_prev)
            print("Y_prev --> size:", len(Y_prev), ", f(Y_prev):", f_on_Y_prev)
            print("obj val local optimum:", str(f_on_X_prev))
        return X_prev

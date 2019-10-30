import warnings
from typing import Set, TypeVar

import numpy as np

from .abstract_optimizer import AbstractOptimizer, AbstractSubmodularFunction

E = TypeVar('E')


class AbstractDoubleGreedySearch(AbstractOptimizer):
    """
    Parent class for Deterministic and
            Randomized (Double Greedy) Unconstrained submodular maximization, by Buchbinder and Feldman

    See also:
        Buchbinder, N., Feldman, M., Naor, J. S., & Schwartz, R. (2015).
        A tight linear time (1/2)-approximation for unconstrained submodular maximization.
        SIAM Journal on Computing, 44(5), 1384–1402. https://doi.org/10.1137/130929205

    """

    def __init__(self, objective_function: AbstractSubmodularFunction, ground_set: Set[E], debug: bool = True):
        super().__init__(objective_function, ground_set, debug)

        self.class_name = "submodmax.AbstractDoubleGreedySearch"

    def should_update_X(self, a: float, b: float):
        raise NotImplementedError('abstract method')

    def optimize(self) -> Set[E]:
        if self.debug:
            print("======================================================")
            print("START", self.class_name)
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

            if self.debug:
                print()
                print("element ", i, "/", ground_set_size)
                print("\t X_prev   --> size: ", len(X_prev), ", f(S):", f_on_Y_prev)
                print("\t X" + str(i) + " + e" + str(i) + " --> size: ", len(X_prev_plus_elem), ", f(S):",
                      f_on_X_prev_plus_elem)

                print("\t Y_prev   --> size: ", len(Y_prev), ", f(S):", f_on_Y_prev)
                print("\t Y" + str(i) + " - e" + str(i) + " --> size: ", len(Y_prev_minus_elem), ", f(S):", f_on_Y_prev_minus_elem)

            if self.should_update_X(a, b):
                X_prev = X_prev_plus_elem
                f_on_X_prev = f_on_X_prev_plus_elem
                # Y_prev stays the same
                if self.debug:
                    print("\tX_prev --> size:", len(X_prev), ", f(X_prev):", f_on_X_prev)
            else:
                # X_prev stays the same
                Y_prev = Y_prev_minus_elem
                f_on_Y_prev = f_on_Y_prev_minus_elem
                if self.debug:
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

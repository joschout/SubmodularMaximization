from typing import Set, TypeVar

from .abstract_optimizer import AbstractOptimizer, AbstractSubmodularFunction

E = TypeVar('E')


class DeterministicDoubleGreedySearch(AbstractOptimizer):
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

    def optimize(self) -> Set[E]:
        X_prev: Set[E] = set()
        Y_prev: Set[E] = self.ground_set

        f_on_X_prev: float = self.objective_function.evaluate(X_prev)
        f_on_Y_prev: float = self.objective_function.evaluate(Y_prev)

        elem: E
        for elem in self.ground_set:

            X_prev_plus_elem: Set[E] = X_prev | {elem}
            f_on_X_prev_plus_elem: float = self.objective_function.evaluate(X_prev_plus_elem)
            a: float = f_on_X_prev_plus_elem - f_on_X_prev

            Y_prev_minus_elem: Set[E] = Y_prev - {elem}
            f_on_Y_prev_minus_elem: float = self.objective_function.evaluate(Y_prev_minus_elem)
            b: float = f_on_Y_prev_minus_elem - f_on_Y_prev

            if a >= b:
                X_prev = X_prev_plus_elem
                f_on_X_prev = f_on_X_prev_plus_elem
                # Y_prev stays the same
            else:
                # X_prev stays the same
                Y_prev = Y_prev_minus_elem
                f_on_Y_prev = f_on_Y_prev_minus_elem

        if not X_prev == Y_prev:
            raise Exception("both sets should be equal")

        if self.debug:
            print("obj val local optimum:", str(f_on_X_prev))
        return X_prev

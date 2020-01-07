import random
from typing import Set, TypeVar, List, Optional, Iterable

from .double_greedy_search_decision_strategy import RandomizedDoubleGreedySearchDecisionStrategy
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
        self.decision_strategy = RandomizedDoubleGreedySearchDecisionStrategy()

        self.n_tries = 10

    def should_update_X(self, a: float, b: float):
        return self.decision_strategy.should_update_X(a, b, self.debug)

    def ground_set_iterator(self) -> Iterable[E]:
        ground_set_list: List[E] = list(self.ground_set)
        random.shuffle(ground_set_list)
        return ground_set_list

    def optimize(self) -> Set[E]:
        if self.n_tries < 1:
            raise Exception(str(self.__class__), " should have self.n_tries >= 1, but has", str(self.n_tries))

        best_set: Optional[Set[E]] = None
        f_best_set = float('-inf')

        for i in range(self.n_tries):
            if self.debug:
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print("START try", str(i+1), "/", str(self.n_tries), self.class_name, 'optimizer')
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")

            current_solution_set: Set[E] = super(RandomizedDoubleGreedySearch, self).optimize()
            f_current_solution_set: float = self.objective_function.evaluate(current_solution_set)

            if f_current_solution_set > f_best_set:
                if self.debug:
                    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    print("Found solution with a higher f-value:", str(f_current_solution_set), ">", str(f_best_set))
                    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                f_best_set = f_current_solution_set
                best_set = current_solution_set
            else:
                if self.debug:
                    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    print("Keep previous solution with f-value:", str(f_best_set))
                    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        if best_set is None:
            raise Exception("best set should not be None")

        return best_set

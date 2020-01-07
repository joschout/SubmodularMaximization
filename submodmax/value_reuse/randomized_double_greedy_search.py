import random
from typing import Set, TypeVar, Iterable, List, Optional, Tuple

from submodmax.double_greedy_search_decision_strategy import RandomizedDoubleGreedySearchDecisionStrategy
from submodmax.value_reuse.abstract_double_greedy_search import AbstractDoubleGreedySearchValueReuse
from submodmax.value_reuse.abstract_optimizer import AbstractSubmodularFunctionValueReuse, FuncInfo
from submodmax.value_reuse.set_info import SetInfo

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

        self.n_tries = 10

    def should_update_X(self, a: float, b: float):
        return self.decision_strategy.should_update_X(a, b, self.debug)

    def ground_set_iterator(self) -> Iterable[E]:
        ground_set_list: List[E] = list(self.ground_set)
        random.shuffle(ground_set_list)
        return ground_set_list

    def optimize(self) -> Tuple[SetInfo, FuncInfo]:
        if self.n_tries < 1:
            raise Exception(str(self.__class__), " should have self.n_tries >= 1, but has", str(self.n_tries))

        best_set_info: Optional[SetInfo] = None
        func_info_best_set: Optional[FuncInfo] = None

        for i in range(self.n_tries):
            if self.debug:
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print("START try", str(i+1), "/", str(self.n_tries), self.class_name, 'optimizer')
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")

            current_rule_set_info: SetInfo
            current_obj_func_val_info: FuncInfo
            current_rule_set_info, current_obj_func_val_info = super(RandomizedDoubleGreedySearch, self).optimize()

            if best_set_info is None:
                best_set_info = current_rule_set_info
                func_info_best_set = current_obj_func_val_info
            elif current_obj_func_val_info.func_value > func_info_best_set.func_value:
                if self.debug:
                    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    print("Found solution with a higher f-value:",
                          str(current_obj_func_val_info.func_value), ">", str(func_info_best_set.func_value))
                    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                best_set_info = current_rule_set_info
                func_info_best_set = current_obj_func_val_info
            else:
                if self.debug:
                    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    print("Keep previous solution with f-value:", str(func_info_best_set.func_value))
                    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        if best_set_info is None:
            raise Exception("best set should not be None")

        return best_set_info, func_info_best_set


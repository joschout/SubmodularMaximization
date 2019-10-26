from typing import Set, TypeVar, Tuple

from value_reuse.abstract_optimizer import AbstractSubmodularFunctionValueReuse, AbstractOptimizerValueReuse, FuncInfo
from value_reuse.set_info import SetInfo

E = TypeVar('E')


class DeterministicDoubleGreedySearch(AbstractOptimizerValueReuse):
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

    def optimize(self) -> Tuple[SetInfo, FuncInfo]:
        if self.debug:
            print("=====================================================================")
            print("START submodmax.value_reuse.DeterministicDoubleGreedySearch optimizer")
            print("=====================================================================")
        # X_prev: Set[E] = set()
        # Y_prev: Set[E] = self.ground_set.copy(

        ground_set_size = len(self.ground_set)
        empty_set = set()


        X_prev_set_info: SetInfo = SetInfo(
            ground_set_size=ground_set_size,
            current_set_size=0,
            added_elems=,
            deleted_elems=,
            intersection_previous_and_current_elems=
        )
        X_prev_set_info.set_current_set(set())
        
        Y_prev_set_info = SetInfo(
            ground_set_size=ground_set_size,
            current_set_size=ground_set_size,
            added_elems=,
            deleted_elems=,
            intersection_previous_and_current_elems=
        )
        Y_prev_set_info.set_current_set(self.ground_set.copy())
        

        func_info_X_prev: FuncInfo = self.objective_function.evaluate(X_prev_set_info, previous_func_info=None)
        func_info_Y_prev: FuncInfo = self.objective_function.evaluate(Y_prev_set_info, previous_func_info=None)

        elem: E
        for elem in self.ground_set:

            singleton_set = {elem}
            
            # X_prev_plus_elem: Set[E] = X_prev | {elem}
            X_prev_plus_elem_set_info = SetInfo(
                ground_set_size=ground_set_size,
                current_set_size=X_prev_set_info.current_set_size + 1,
                added_elems=singleton_set,
                deleted_elems=empty_set,
                intersection_previous_and_current_elems= X_prev_set_info.current_set
            )

            func_info_X_prev_plus_elem: FuncInfo = self.objective_function.evaluate(
                X_prev_plus_elem_set_info, func_info_X_prev)
            a: float = func_info_X_prev_plus_elem.func_value - func_info_X_prev.func_value

            Y_prev_minus_elem_set: Set[E] = Y_prev_set_info.current_set - {elem}
            Y_prev_minus_elem_set_info = SetInfo(
                ground_set_size=ground_set_size,
                current_set_size=Y_prev_set_info.current_set_size - 1,
                added_elems=empty_set,
                deleted_elems=singleton_set,
                intersection_previous_and_current_elems=Y_prev_minus_elem_set
            )
            func_info_Y_prev_minus_elem = self.objective_function.evaluate(
                Y_prev_minus_elem_set_info, func_info_Y_prev)
            b: float = func_info_Y_prev_minus_elem.func_value - func_info_Y_prev.func_value

            if a >= b:

                new_set = X_prev_set_info.current_set | singleton_set
                X_prev_plus_elem_set_info.set_current_set(new_set)
                X_prev_set_info = X_prev_plus_elem_set_info

                func_info_X_prev = func_info_X_prev_plus_elem
                # Y_prev stays the same
            else:
                # X_prev stays the same
                Y_prev = Y_prev_minus_elem_set_info
                func_info_Y_prev = func_info_Y_prev_minus_elem

        if not X_prev_set_info.current_set == Y_prev_set_info.current_set:
            raise Exception("both sets should be equal")

        if self.debug:
            print("obj val local optimum:", str(func_info_X_prev.func_value))
        return X_prev_set_info, func_info_X_prev

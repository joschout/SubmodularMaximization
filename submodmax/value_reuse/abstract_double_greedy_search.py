import warnings
from typing import Set, TypeVar, Tuple

from value_reuse.abstract_optimizer import AbstractSubmodularFunctionValueReuse, AbstractOptimizerValueReuse, FuncInfo
from value_reuse.set_info import SetInfo

E = TypeVar('E')


class AbstractDoubleGreedySearchValueReuse(AbstractOptimizerValueReuse):
    """
    Parent class for Deterministic and
        Randomized  Unconstrained submodular maximization, by Buchbinder and Feldman

    See also:
        Buchbinder, N., Feldman, M., Naor, J. S., & Schwartz, R. (2015).
        A tight linear time (1/2)-approximation for unconstrained submodular maximization.
        SIAM Journal on Computing, 44(5), 1384–1402. https://doi.org/10.1137/130929205

    """

    def __init__(self, objective_function: AbstractSubmodularFunctionValueReuse, ground_set: Set[E],
                 debug: bool = True):
        super().__init__(objective_function, ground_set, debug)

        self.class_name = 'submodmax.value_reuse.AbstractDoubleGreedySearch'

    def should_update_X(self, a: float, b: float) -> bool:
        raise NotImplementedError('abstract method')

    def optimize(self) -> Tuple[SetInfo, FuncInfo]:
        if self.debug:
            print("=====================================================================")
            print("START", self.class_name, "optimizer")
            print("=====================================================================")
        ground_set_size = len(self.ground_set)
        empty_set = set()

        X_prev_set_info: SetInfo = SetInfo(
            ground_set_size=ground_set_size,
            current_set_size=0,
            added_elems=empty_set,
            deleted_elems=empty_set,
            intersection_previous_and_current_elems=empty_set
        )
        X_prev_set_info.set_current_set(set())

        Y_prev_set_info = SetInfo(
            ground_set_size=ground_set_size,
            current_set_size=ground_set_size,
            added_elems=self.ground_set,
            deleted_elems=empty_set,
            intersection_previous_and_current_elems=empty_set
        )
        Y_prev_set_info.set_current_set(self.ground_set.copy())

        func_info_X_prev: FuncInfo = self.objective_function.evaluate(X_prev_set_info, previous_func_info=None)
        func_info_Y_prev: FuncInfo = self.objective_function.evaluate(Y_prev_set_info, previous_func_info=None)

        if self.debug:
            print("initialization:")
            print("X0 : size: ", X_prev_set_info.current_set_size, "/", ground_set_size, ", f(S): ",
                  func_info_X_prev.func_value)
            print("Y0:  size: ", Y_prev_set_info.current_set_size, "/", ground_set_size, ", f(S): ",
                  func_info_Y_prev.func_value)

        elem: E
        for i, elem in enumerate(self.ground_set, 1):

            singleton_set: Set[E] = {elem}

            # X_prev_plus_elem: Set[E] = X_prev | {elem}
            X_prev_plus_elem_set_info = SetInfo(
                ground_set_size=ground_set_size,
                current_set_size=X_prev_set_info.current_set_size + 1,
                added_elems=singleton_set,
                deleted_elems=empty_set,
                intersection_previous_and_current_elems=X_prev_set_info.current_set
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
            Y_prev_minus_elem_set_info.set_current_set(Y_prev_minus_elem_set)
            func_info_Y_prev_minus_elem = self.objective_function.evaluate(
                Y_prev_minus_elem_set_info, func_info_Y_prev)
            b: float = func_info_Y_prev_minus_elem.func_value - func_info_Y_prev.func_value

            if self.debug:
                print()
                print("element ", i, "/", ground_set_size)
                print("\t X_prev   --> size: ", X_prev_set_info.current_set_size, ", f(S):",
                      func_info_X_prev.func_value)
                print("\t X" + str(i) + " + e" + str(i) + " --> size: ", X_prev_plus_elem_set_info.current_set_size,
                      ", f(S):",
                      func_info_X_prev_plus_elem.func_value)
                print()
                print("\t Y_prev   --> size: ", Y_prev_set_info.current_set_size, ", f(S):",
                      func_info_Y_prev.func_value)
                print("\t Y" + str(i) + " - e" + str(i) + " --> size: ", Y_prev_minus_elem_set_info.current_set_size,
                      ", f(S):",
                      func_info_Y_prev_minus_elem.func_value)

            if self.should_update_X(a, b):

                new_set = X_prev_set_info.current_set | singleton_set
                X_prev_plus_elem_set_info.set_current_set(new_set)
                X_prev_set_info = X_prev_plus_elem_set_info

                func_info_X_prev = func_info_X_prev_plus_elem
                # Y_prev stays the same
                if self.debug:
                    print("\tX_prev --> size:", X_prev_set_info.current_set_size, ", f(X_prev):",
                          func_info_X_prev.func_value)
            else:
                # X_prev stays the same
                Y_prev_set_info = Y_prev_minus_elem_set_info
                func_info_Y_prev = func_info_Y_prev_minus_elem
                if self.debug:
                    print("\tY_prev --> size:", Y_prev_set_info.current_set_size, ", f(Y_prev):",
                          func_info_Y_prev.func_value)

        warnings.warn("remove equality check")
        if not X_prev_set_info.current_set == Y_prev_set_info.current_set:
            raise Exception("both sets should be equal")

        if self.debug:
            print("-- finished iteration --")
            print("X_prev --> size:", X_prev_set_info.current_set_size, ", f(X_prev):", func_info_X_prev.func_value)
            print("Y_prev --> size:", Y_prev_set_info.current_set_size, ", f(Y_prev):", func_info_Y_prev.func_value)
            print("obj val local optimum:", str(func_info_X_prev.func_value))
        return X_prev_set_info, func_info_X_prev

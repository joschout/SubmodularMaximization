from typing import Set, Tuple, Optional, TypeVar

from .abstract_optimizer import AbstractSubmodularFunctionValueReuse, AbstractOptimizerValueReuse, FuncInfo
from .set_info import SetInfo

E = TypeVar('E')


class DeterministicLocalSearchValueReuse(AbstractOptimizerValueReuse):
    """

    Goal: find a set S \\subset X that maximizes a (possibly) non-monoton submodular function f : 2 -> R+

    Increase the value of our solution by
        either adding a new element in S
        of discarding one of the elements in S.
    S is a local optimum if no such operation increases the value of S

    Guarantee: if this terminates, the found set S is a (1 -  epsilon/|X|^2)-approximate local optimum:
        adding or removing an element changes the objective function value with at most a factor (1 -  epsilon/|X|^2)

    It is a (1/3 -  epsilon/|X|)-approximation algorithm: either the found solution set S or its complement X - S
        have an objective function value that differs from the global optimum
         with at most a factor (1/3 -  epsilon/|X|).

    step by step implementation of deterministic local search algorithm in the
    FOCS paper: https://people.csail.mit.edu/mirrokni/focs07.pdf (page 4-5)
    """

    def __init__(self, objective_function: AbstractSubmodularFunctionValueReuse, ground_set: Set[E],
                 epsilon: float = 0.05,
                 debug: bool = True):
        super().__init__(objective_function, ground_set, debug)
        self.epsilon: float = epsilon
        n: int = len(ground_set)
        self.rho: float = (1 + self.epsilon / (n * n))

        self.empty_set = set()

    def optimize(self) -> Tuple[SetInfo, FuncInfo]:
        if self.debug:
            print("==================================================")
            print("START DeterministicLocalSearchValueReuse optimizer")
            print("==================================================")

        solution_set_info: SetInfo
        func_info1: FuncInfo

        solution_set_info, func_info1 \
            = self._deterministic_local_search()

        complement_set: Set[E] = self.ground_set - solution_set_info.current_set
        complement_set_info = SetInfo(ground_set_size=len(self.ground_set),
                                      current_set_size=len(complement_set),
                                      added_elems=complement_set,
                                      deleted_elems=self.empty_set,
                                      intersection_previous_and_current_elems=self.empty_set
                                      )
        func_info2: FuncInfo = self.objective_function.evaluate(complement_set_info,
                                                                previous_func_info=None)

        if func_info1.func_value >= func_info2.func_value:
            if self.debug:
                print("Choosing first set as solution.")
                print("Objective value of chosen solution set:", func_info1.func_value)
                print("Chosen solution set size:", solution_set_info.current_set_size, " / ", len(self.ground_set))
                print("Objective value of other set:", func_info2.func_value)
                print("Other set size:", complement_set_info.current_set_size, " / ", len(self.ground_set))

            return solution_set_info, func_info1
        else:
            if self.debug:
                print("Choosing second set as solution.")
                print("Objective value of chosen solution set:", func_info2.func_value)
                print("Chosen solution set size:", complement_set_info.current_set_size, " / ", len(self.ground_set))
                print("Objective value of other set:", func_info1.func_value)
                print("Other set size:", solution_set_info.current_set_size, " / ", len(self.ground_set))

            return complement_set_info, func_info2

    def _get_initial_subset_and_objective_function_value(self) -> Tuple[SetInfo, FuncInfo]:
        # the initial subset is the maximum over all singletons v in X

        best_singleton_set_info: Optional[SetInfo] = None
        best_func_info: Optional[FuncInfo] = None

        for elem in self.ground_set:
            new_singleton_set = {elem}
            new_singleton_set_info = SetInfo(ground_set_size=len(self.ground_set),
                                             current_set_size=1,
                                             added_elems=new_singleton_set,
                                             deleted_elems=self.empty_set,
                                             intersection_previous_and_current_elems=self.empty_set
                                             )

            func_info: FuncInfo = self.objective_function.evaluate(new_singleton_set_info,
                                                                   previous_func_info=None)
            if best_func_info is None:
                best_func_info = func_info
                new_singleton_set_info.set_current_set(new_singleton_set)
                best_singleton_set_info = new_singleton_set_info

            else:
                if func_info.func_value > best_func_info.func_value:
                    best_func_info = func_info
                    new_singleton_set_info.current_set = new_singleton_set
                    best_singleton_set_info = new_singleton_set_info
                else:
                    pass

        if best_singleton_set_info is None:
            raise Exception("no good initial subset found")
        if best_func_info is None:
            raise Exception("you should not be able to reach this")

        if self.debug:
            print("Initial solution set size:", best_singleton_set_info.current_set_size)
            print("Current solution set:")
            for i, elem in enumerate(best_singleton_set_info.current_set, 1):
                print("\t", i, elem)
            print("func val:", best_func_info.func_value)

        return best_singleton_set_info, best_func_info

    def _deterministic_local_search(self) -> Tuple[SetInfo, FuncInfo]:
        # the initial subset is the maximum over all singletons v in X

        solution_set_info: SetInfo
        solution_set_func_info: FuncInfo
        solution_set_info, solution_set_func_info = self._get_initial_subset_and_objective_function_value()

        # Increase the value of our solution by
        #         either adding a new element in S
        #         or discarding one of the elements in S.
        #
        # S is a local optimum if no such operation increases the value of S
        local_optimum_found: bool = False
        while not local_optimum_found:
            # keep adding elements until there is no improvement

            size_before_adding = solution_set_info.current_set_size
            solution_set_info, solution_set_func_info = self._add_elements_until_no_improvement(
                solution_set_info, solution_set_func_info)
            size_after_adding = solution_set_info.current_set_size
            print("Added", size_after_adding - size_before_adding, "elements", "(Current size:", size_after_adding, ")", "WITH val reuse")

            # check if removing an element leads to improvement
            #   if it does, restart with adding elements
            #   if it does not, we have found a local optimum
            an_improved_subset: Optional[Tuple[SetInfo, FuncInfo]] = \
                self._find_improved_subset_by_discarding_one_element(solution_set_info, solution_set_func_info)
            if an_improved_subset is None:
                local_optimum_found = True
            else:
                solution_set_info, solution_set_func_info = an_improved_subset
                size_after_deleting = solution_set_info.current_set_size
                print("Deleted ", size_after_adding - size_after_deleting, "elements", "(Current size:", size_after_deleting, ")", "WITH val reuse")

        return solution_set_info, solution_set_func_info

    def _add_elements_until_no_improvement(self,
                                           current_set_info: SetInfo,
                                           current_func_info: FuncInfo) -> Tuple[SetInfo, FuncInfo]:
        """
        Keep adding elements to the solution set,
            as long as adding an element increases the submodular (objective) function with (1 + epsilon / (n * n))
        """
        solution_set_info: SetInfo = current_set_info
        solution_set_func_info: FuncInfo = current_func_info

        improvement_possible_by_adding: bool = True
        while improvement_possible_by_adding:

            if current_set_info.current_set is None:
                raise Exception("Warning: this should never happen!")

            an_improved_subset: Optional[Tuple[SetInfo, FuncInfo]] = self._find_improved_subset_by_adding_one_element(
                solution_set_info, solution_set_func_info)
            if an_improved_subset is not None:
                solution_set_info, solution_set_func_info = an_improved_subset
            else:
                improvement_possible_by_adding = False
        return solution_set_info, solution_set_func_info

    def _find_improved_subset_by_adding_one_element(self,
                                                    solution_set_info: SetInfo,
                                                    solution_set_func_info: FuncInfo) \
            -> Optional[Tuple[SetInfo, FuncInfo]]:
        """
        Find AN element that when added to the current subset,
        increases the value of the submodular function with (1 + epsilon / (n * n)):
         f(S+e) - f(S) > (1 + epsilon / (n * n))

        """
        # -- Adding elements to S
        # Keep adding elements to S so long as the improvement is larger than (1 + epsilon)/ n²
        # i.e. f(S + e) - f(S) > (1 + epsilon)/ n²

        mod_solution_set_info = SetInfo(
            ground_set_size=solution_set_info.ground_set_size,
            current_set_size=solution_set_info.current_set_size + 1,
            # current_set=mod_solution_set,
            added_elems=None,
            deleted_elems=self.empty_set,
            intersection_previous_and_current_elems=solution_set_info.current_set
        )

        elem: E
        for elem in self.ground_set:
            if elem in solution_set_info.current_set:
                continue

            if self.debug:
                print("\tTesting if elem is good to add " + str(elem))
            added_elem: Set[E] = {elem}

            mod_solution_set_info.added_elems = added_elem
            func_info_mod_solution_set: FuncInfo = self.objective_function.evaluate(mod_solution_set_info,
                                                                                    solution_set_func_info)
            diff = func_info_mod_solution_set.func_value - self.rho * solution_set_func_info.func_value

            # print("\tfunc val mod set:", func_info_mod_solution_set.func_value)
            # print("\tfunc val unmod  :", solution_set_func_info.func_value)
            # print("\tdiff:", diff)

            if func_info_mod_solution_set.func_value > self.rho * solution_set_func_info.func_value:
                if self.debug:
                    print("-----------------------")
                    print("Adding to the solution set elem " + str(elem))
                    print("-----------------------")
                mod_solution_set: Set[E] = solution_set_info.current_set | added_elem
                # mod_solution_set_info.current_set = mod_solution_set
                mod_solution_set_info.current_set = mod_solution_set

                if mod_solution_set_info.current_set is None:
                    raise Exception("uninitialized current set")

                return mod_solution_set_info, func_info_mod_solution_set
                # return mod_solution_set_info, func_info_mod_solution_set
        # None of the remaining elements increase the objective function with more than (1 + epsilon / (n * n))
        # if mod_solution_set_info.current_set is None:
        #     raise Exception("uninitialized current set")

        return None

    def _find_improved_subset_by_discarding_one_element(self,
                                                        solution_set_info: SetInfo,
                                                        solution_set_func_info: FuncInfo) \
            -> Optional[Tuple[SetInfo, FuncInfo]]:

        mod_solution_set_info = SetInfo(
            ground_set_size=len(self.ground_set),
            current_set_size=solution_set_info.current_set_size - 1,
            added_elems=self.empty_set,
            deleted_elems=None,
            intersection_previous_and_current_elems=None)

        mod_solution_set: Set[E] = solution_set_info.current_set.copy()

        for elem in solution_set_info.current_set:
            if self.debug:
                print("\tTesting should remove elem " + str(elem))

            deleted_elem = {elem}

            mod_solution_set_info.deleted_elems = deleted_elem

            mod_solution_set.remove(elem)
            mod_solution_set_info.intersection_previous_and_current_elems = mod_solution_set
            func_info_mod_solution_set: FuncInfo = self.objective_function.evaluate(mod_solution_set_info,
                                                                                    solution_set_func_info)

            if func_info_mod_solution_set.func_value > self.rho * solution_set_func_info.func_value:
                if self.debug:
                    print("-----------------------")
                    print("Removing from solution set elem " + str(elem))
                    print("-----------------------")
                mod_solution_set_info.current_set = mod_solution_set
                return mod_solution_set_info, func_info_mod_solution_set
            else:
                mod_solution_set.add(elem)
        # None of there is no element that when removed
        #   increases the objective function with more than (1 + epsilon / (n * n))
        return None

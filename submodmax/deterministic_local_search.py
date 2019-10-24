from typing import Set, Tuple, Optional, TypeVar

from .abstract_optimizer import AbstractOptimizer, AbstractSubmodularFunction

E = TypeVar('E')


class DeterministicLocalSearch(AbstractOptimizer):
    """

    Goal: find a set S subset X that maximizes a (possibly) non-monoton submodular function f : 2 -> R+

    Increase the value of our solution by
        either adding a new element in S
        of discarding one of the elements in S.
    S is a local optimum if no such operation increases the value of S

    Guarantee: if this terminates, the found set S is a (1 -  epsilon/|X|^2)-approximate local optimum:
        adding or removing an element changes the objective function value with at most a factor (1 -  epsilon/|X|^2)

    It is a (1/3 -  epsilon/|X|)-approximation algorithm:
        either the found solution set S or its complement X - S
        have an objective function value that differs from the global optimum
        with at most a factor (1/3 -  epsilon/|X|).

    step by step implementation of deterministic local search algorithm in the
    FOCS paper: https://people.csail.mit.edu/mirrokni/focs07.pdf (page 4-5)
    """

    def __init__(self, objective_function: AbstractSubmodularFunction, ground_set: Set[E], epsilon: float = 0.05,
                 debug: bool = True):
        super().__init__(objective_function, ground_set, debug)
        self.epsilon: float = epsilon
        n: int = len(ground_set)
        self.rho: float = 0.5 * (1 + self.epsilon / (n * n))

    def optimize(self) -> Set[E]:
        solution_set, func_val1 = self._deterministic_local_search()  # type: Set[E], float
        complement_of_solution_set: Set[E] = self.ground_set - solution_set
        func_val2: float = self.objective_function.evaluate(complement_of_solution_set)

        if func_val1 >= func_val2:
            if self.debug:
                print("Objective value of solution set:", func_val1)
            return solution_set
        else:
            if self.debug:
                print("Objective value of solution set:", func_val2)
            return complement_of_solution_set

    def _get_initial_subset_and_objective_function_value(self) -> Tuple[Set[E], float]:
        # the initial subset is the maximum over all singletons v in X

        best_singleton_set: Optional[Set[E]] = None
        best_func_val = float('-inf')

        elem: E
        for elem in self.ground_set:
            new_singleton_set: Set[E] = {elem}
            func_val: float = self.objective_function.evaluate(new_singleton_set)
            if func_val > best_func_val:
                best_singleton_set = new_singleton_set
                best_func_val = func_val

        if best_singleton_set is None:
            raise Exception("no good initial subset found")

        return best_singleton_set, best_func_val

    def _find_improved_subset_by_adding_one_element(self, solution_set: Set[E],
                                                    func_val_solution_set: float) -> Optional[Tuple[Set[E], float]]:
        """
        Find AN element that when added to the current subset,
        increases the value of the submodular function with (1 + epsilon / (n * n)):
         f(S+e) - f(S) > (1 + epsilon / (n * n))

        """
        # -- Adding elements to S
        # Keep adding elements to S so long as the improvement is larger than (1 + epsilon)/ n²
        # i.e. f(S + e) - f(S) > (1 + epsilon)/ n²
        #
        # possible_items_to_add: Set[E] = self.ground_set - solution_set
        elem: E
        for elem in self.ground_set:
            if elem not in solution_set:
                if self.debug:
                    print("Testing if elem is good to add " + str(elem))

                mod_solution_set: Set[E] = solution_set | {elem}
                func_val_mod_solution_set: float = self.objective_function.evaluate(mod_solution_set)

                diff = abs(func_val_mod_solution_set - self.rho * func_val_solution_set)

                # print("\tfunc val mod set:", func_val_mod_solution_set)
                # print("\tfunc val unmod  :", func_val_solution_set)
                # print("\tdiff:", diff)

                if func_val_mod_solution_set > self.rho * func_val_solution_set:
                    if self.debug:
                        print("-----------------------")
                        print("Adding to the solution set elem " + str(elem))
                        print("-----------------------")
                    return mod_solution_set, func_val_mod_solution_set
        # None of the remaining elements increase the objective function with more than (1 + epsilon / (n * n))
        return None

    def _find_improved_subset_by_discarding_one_element(self, solution_set: Set[E], func_val_solution_set: float):
        # possible_items_to_remove = solution_set
        elem: E
        for elem in solution_set:
            if self.debug:
                print("Testing should remove elem " + str(elem))

            mod_solution_set: Set[E] = solution_set - {elem}
            func_val_mod_solution_set = self.objective_function.evaluate(mod_solution_set)

            if func_val_mod_solution_set > self.rho * func_val_solution_set:
                if self.debug:
                    print("-----------------------")
                    print("Removing from solution set elem " + str(elem))
                    print("-----------------------")

                return mod_solution_set, func_val_mod_solution_set
        # Return None if there is no element that when removed
        #   increases the objective function with more than (1 + epsilon / (n * n))
        return None

    def _deterministic_local_search(self) -> Tuple[Set[E], float]:
        # the initial subset is the maximum over all singletons v in X
        solution_set: Set[E]
        solution_set_obj_func_val: float
        solution_set, solution_set_obj_func_val = self._get_initial_subset_and_objective_function_value()

        # Increase the value of our solution by
        #         either adding a new element in S
        #         or discarding one of the elements in S.
        #
        # S is a local optimum if no such operation increases the value of S
        local_optimum_found: bool = False
        while not local_optimum_found:
            # keep adding elements until there is no improvement

            size_before_adding = len(solution_set)
            solution_set, solution_set_obj_func_val = self._add_elements_until_no_improvement(
                solution_set, solution_set_obj_func_val)
            size_after_adding = len(solution_set)
            print("Added", size_after_adding - size_before_adding, "elements", "(Current size:", size_after_adding, ")", "NO val reuse")

            # check if removing an element leads to improvement
            #   if it does, restart with adding elements
            #   if it does not, we have found a local optimum
            an_improved_subset: Optional[Tuple[Set[E]], float] = self._find_improved_subset_by_discarding_one_element(
                solution_set, solution_set_obj_func_val)
            if an_improved_subset is None:
                local_optimum_found = True
            else:
                solution_set, solution_set_obj_func_val = an_improved_subset
                size_after_deleting = len(solution_set)
                print("Deleted ", size_after_adding - size_after_deleting, "elements.", "(Current size:", size_after_deleting, ")", "NO val reuse")

        return solution_set, solution_set_obj_func_val

    def _add_elements_until_no_improvement(self, current_solution_set: Set[E],
                                           current_obj_func_val: float) -> Tuple[Set[E], float]:
        """
        Keep adding elements to the solution set,
            as long as adding an element increases the submodular (objective) function with (1 + epsilon / (n * n))
        """
        solution_set: Set[E] = current_solution_set
        solution_set_obj_func_val: float = current_obj_func_val

        improvement_possible_by_adding: bool = True
        while improvement_possible_by_adding:
            an_improved_subset: Optional[Tuple[Set[E]], float] = self._find_improved_subset_by_adding_one_element(
                solution_set, solution_set_obj_func_val)
            if an_improved_subset is not None:
                solution_set, solution_set_obj_func_val = an_improved_subset
            else:
                improvement_possible_by_adding = False
        return solution_set, solution_set_obj_func_val

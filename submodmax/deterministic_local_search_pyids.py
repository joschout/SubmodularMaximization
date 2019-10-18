from typing import Set, Tuple, Optional, TypeVar

from abstract_optimizer import AbstractOptimizer, AbstractObjectiveFunction

E = TypeVar('E')


class DeterministicLocalSearchPyIDS(AbstractOptimizer):
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

    This implementation is largely based on the one from Jiri Filip and Tomas Kliegr included in PyIDS.
    """

    def __init__(self, objective_function: AbstractObjectiveFunction, ground_set: Set[E], epsilon=0.05,
                 debug: bool = True):
        super().__init__(objective_function, ground_set, debug)
        self.epsilon: float = epsilon
        n: int = len(ground_set)
        self.rho: float = (1 + self.epsilon / (n * n))

    def optimize(self) -> Set[E]:
        solution_set: Set[E] = self._deterministic_local_search()
        complement_of_solution_set: Set[E] = self.ground_set - solution_set

        func_val1 = self.objective_function.evaluate(solution_set)
        func_val2 = self.objective_function.evaluate(complement_of_solution_set)

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

    def _deterministic_local_search(self) -> Set[E]:
        # the initial subset is the maximum over all singletons v in X
        solution_set: Set[E]
        soln_set_obj_func_value: float
        solution_set, soln_set_obj_func_value = self._get_initial_subset_and_objective_function_value()

        restart_computations: bool = False

        while True:

            # -- Adding elements to S
            # Keep adding elements to S so long as the improvement is larger than (1 + epsilon)/ n²
            # i.e. f(S + e) - f(S) > (1 + epsilon)/ n²
            elem: E
            for elem in self.ground_set - solution_set:
                if self.debug:
                    print("Testing if elem is good to add " + str(elem))

                modified_solution_set: Set[E] = solution_set | {elem}
                func_val: float = self.objective_function.evaluate(modified_solution_set)

                if func_val > self.rho * soln_set_obj_func_value:
                    # add this element to solution set and recompute omegas
                    solution_set.add(elem)
                    soln_set_obj_func_value = func_val
                    restart_computations = True

                    if self.debug:
                        print("-----------------------")
                        print("Adding to the solution set elem:  " + str(elem))
                        print("-----------------------")
                    break

            # ----------------

            if restart_computations:
                restart_computations = False
                continue

            # --- Discarding elements of S ---

            elem: E
            for elem in solution_set:
                if self.debug:
                    print("Testing should remove elem " + str(elem))

                modified_solution_set: Set[E] = solution_set - {elem}
                func_val: float = self.objective_function.evaluate(modified_solution_set)

                if func_val > self.rho * soln_set_obj_func_value:
                    # add this element to solution set and recompute omegas
                    solution_set.add(elem)
                    soln_set_obj_func_value = func_val
                    restart_computations = True

                    if self.debug:
                        print("-----------------------")
                        print("Removing from solution set elem " + str(elem))
                        print("-----------------------")
                    break
            # ----------------

            if restart_computations:
                restart_computations = False
                continue

            return solution_set

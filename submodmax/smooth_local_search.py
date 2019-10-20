import math
import warnings
from typing import Set, Optional, TypeVar, List

import numpy as np

from .abstract_optimizer import AbstractOptimizer, AbstractSubmodularFunction
from .random_set import sample_a_set_with_bias_delta_on_A

E = TypeVar('E')


class SmoothLocalSearch(AbstractOptimizer):
    """
    Optimization procedure
        to find a subset of the larger set X
            that is an approximate solution to the objective function with some theoretical guarantees.


    This requires the objective function to be:
        * non-negative
        * non-normal
        * non-monotone
        * submodular
    over the power set of X.

    Note: the problem of maximizing a submodular function is NP-hard.

    """
    def __init__(self, objective_function: AbstractSubmodularFunction, ground_set: Set[E], debug=True):
        super().__init__(objective_function, ground_set, debug)

        self.empty_set: Set[E] = set()
        self.ground_set_size: int = len(self.ground_set)

    def optimize(self) -> Set[E]:
        delta1: float = 1/3
        delta_prime1: float = 1/3

        delta2: float = 1/3
        delta_prime2: float = -1.0

        solution_set1: Set[E] = self._smooth_local_search(delta=delta1, delta_prime=delta_prime1)
        solution_set2: Set[E] = self._smooth_local_search(delta=delta2, delta_prime=delta_prime2)

        func_val1 = self.objective_function.evaluate(solution_set1)
        func_val2 = self.objective_function.evaluate(solution_set2)

        if func_val1 >= func_val2:
            return solution_set1
        else:
            return solution_set2

    def _smooth_local_search(self, delta: float, delta_prime: float) -> Set[E]:

        # Estimate the OPTIMAL value, i.e. the value of the submodular function for the optimal subset
        OPT: float = self._estimate_optimal_objective_function_value()
        n: int = self.ground_set_size
        omega_threshold: float = 2.0 / (n * n) * OPT
        error_bound_for_estimated_omega: float = 1 / (n * n) * OPT

        if self.debug:
            print("bound for omega estimate 2/(n*n) * OPTIMUM VALUE =", omega_threshold)

        # initialize the solution set with the empty set
        current_solution_set: Set[E] = set()
        current_solution_set_complement: Set[E] = self.ground_set.copy()

        nb_of_adds: int = 0
        nb_of_removes: int = 0

        # For each element in the ground set,calculate the estimate of omega for the current set A
        #   IF an omega_estima > omega_bound:
        #       add elem to A,
        #       restart
        #    ELSE
        #       IF an omega_estimate < - omega_bound
        #          add elem to A
        #          restart
        # S is a local optimum of a smoothed version of the objective function
        local_optimum_found: bool = False
        while not local_optimum_found:

            optional_elem_to_add: Optional[E] = self._find_elem_in_complement_of_solution_set_with_sufficiently_large_omega(
                current_solution_set, current_solution_set_complement,
                error_bound_for_estimated_omega, omega_threshold, delta)

            if optional_elem_to_add is None:
                optional_elem_to_remove: Optional[E] = self._find_elem_in_solution_set_with_sufficiently_low_omega(
                    current_solution_set,
                    error_bound_for_estimated_omega, omega_threshold, delta)
                if optional_elem_to_remove is None:
                    local_optimum_found = True
                else:
                    if self.debug:
                        print("removing elem from solution set", str(optional_elem_to_remove))
                    current_solution_set.remove(optional_elem_to_remove)
                    current_solution_set_complement.add(optional_elem_to_remove)
                    nb_of_removes += 1
            else:
                if self.debug:
                    print("adding elem to solution set:", str(optional_elem_to_add))
                current_solution_set.add(optional_elem_to_add)
                current_solution_set_complement.remove(optional_elem_to_add)
                nb_of_adds += 1
        if self.debug:
            print("nb of elements added:", nb_of_adds)
            print("nb of elements removed:", nb_of_removes)

        if len(current_solution_set) == 0:
            print("the final set based on which is sampled is empty")
            warnings.warn("the final set based on which is sampled is empty")

        return sample_a_set_with_bias_delta_on_A(current_solution_set, self.ground_set, delta_prime)

    def _find_elem_in_complement_of_solution_set_with_sufficiently_large_omega(self,
                                                                               current_solution_set: Set[E],
                                                                               complement_of_current_solution_set: Set[E],
                                                                               error_threshold: float,
                                                                               omega_estimate_threshold: float,
                                                                               delta: float) -> Optional[E]:
        for elem in complement_of_current_solution_set:
            omega_est = self._estimate_omega(elem, current_solution_set, error_threshold, delta)
            if omega_est > omega_estimate_threshold:
                # add this element to solution set and recompute omegas
                return elem
        return None

    def _find_elem_in_solution_set_with_sufficiently_low_omega(self,
                                                               current_solution_set: Set[E],
                                                               error_threshold: float,
                                                               omega_estimate_threshold: float,
                                                               delta: float) -> Optional[E]:
        elem: E
        for elem in current_solution_set:
            omega_est = self._estimate_omega(elem, current_solution_set, error_threshold, delta)
            if omega_est < - omega_estimate_threshold:
                return elem
        return None

    def _estimate_optimal_objective_function_value(self) -> float:
        """
        Estimate the optimal value for the objective function, by
         1. taking a sample subset of the possible elements, each element sampled with probability 1/2,
         2. evaluating the objective function for this random solution set
        """
        uniformly_random_subset: Set[E] = sample_a_set_with_bias_delta_on_A(
            self.empty_set, self.ground_set, delta=0.5)
        estimated_optimal_objective_function_value: float = self.objective_function.evaluate(uniformly_random_subset)
        return estimated_optimal_objective_function_value

    def _estimate_omega(self, elem: E, solution_set: Set[E], error_threshold, delta) -> float:
        if self.debug:
            print("Estimating omega for elem", elem, sep="\n")

        # NOTE: these lists are never emptied/re-initialized
        expected_values_of_function_include_x: List[E] = []
        expected_values_of_function_exclude_x: List[E] = []

        standard_error: float = float('+inf')
        while standard_error > error_threshold:
            # first expectation term (include x), over 10 samples
            for _ in range(10):
                temp_soln_set: Set[E] = sample_a_set_with_bias_delta_on_A(solution_set, self.ground_set, delta)
                temp_soln_set.add(elem)

                func_val: float = self.objective_function.evaluate(temp_soln_set)

                expected_values_of_function_include_x.append(func_val)

            # second expectation term (exclude x), over 10 samples
            for _ in range(10):
                temp_soln_set: Set[E] = sample_a_set_with_bias_delta_on_A(solution_set, self.ground_set, delta)
                if elem in temp_soln_set:
                    temp_soln_set.remove(elem)

                func_val: float = self.objective_function.evaluate(temp_soln_set)

                expected_values_of_function_exclude_x.append(func_val)

            # compute standard error of mean difference
            variance_exp_include: float = np.var(expected_values_of_function_include_x)
            variance_exp_exclude: float = np.var(expected_values_of_function_exclude_x)
            standard_error: float = math.sqrt(
                variance_exp_include / len(expected_values_of_function_include_x) + variance_exp_exclude / len(
                    expected_values_of_function_exclude_x))

            if self.debug:
                print("\tStandard Error:", standard_error, ", Error Threshold:", error_threshold)

        return np.mean(expected_values_of_function_include_x) - np.mean(expected_values_of_function_exclude_x)

import math
import warnings
from typing import TypeVar, Set, List

import numpy as np

from abstract_optimizer import AbstractOptimizer, AbstractObjectiveFunction
from random_set import sample_a_set_with_bias_delta_on_A, RandomSetOptimizer

E = TypeVar('E')


class SmoothLocalSearch(AbstractOptimizer):
    """
    Smooth Local Search optimizer
    """

    def __init__(self, objective_function: AbstractObjectiveFunction, ground_set: Set[E], debug: bool = True):
        super().__init__(objective_function, ground_set, debug)
        self.rs_optimizer = RandomSetOptimizer(ground_set)

    def optimize(self) -> Set[E]:
        delta1: float = 1 / 3
        delta_prime1: float = 1 / 3

        delta2: float = 1 / 3
        delta_prime2: float = -1.0

        solution_set1: Set[E] = self._smooth_local_search(delta=delta1, delta_prime=delta_prime1)
        solution_set2: Set[E] = self._smooth_local_search(delta=delta2, delta_prime=delta_prime2)

        func_val1: float = self.objective_function.evaluate(solution_set1)
        func_val2: float = self.objective_function.evaluate(solution_set2)

        if func_val1 >= func_val2:
            return solution_set1
        else:
            return solution_set2

    def _smooth_local_search(self, delta: float, delta_prime: float) -> Set[E]:
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

        :param delta:
        :param delta_prime:
        :return:
        """

        n = len(self.ground_set)

        # Estimate the OPTIMAL value, i.e. the value of the submodular function for the optimal subset
        OPT: float = self._compute_OPT()
        # initialize the solution set with the empty set
        current_solution_set: Set[E] = set()

        error_threshold: float = 1 / (n * n) * OPT
        estimated_omega_bound: float = 2.0 * error_threshold

        if self.debug:
            print("2/(n*n) * OPTIMUM VALUE =", estimated_omega_bound)

        restart_omega_computations: bool = False

        nb_of_adds: int = 0
        nb_of_removes: int = 0

        while True:
            # step 2 & 3: for each element,
            #       estimate omega within a certain error_threshold;
            #       if estimated omega > 2/n^2 * OPT,
            #       then add the corresponding elem to soln set
            #            and recompute omega estimates again
            omega_estimates: List[float] = []
            elem: E
            for elem in self.ground_set:

                if self.debug:
                    print("Estimating omega for elem", elem, sep="\n")

                warnings.warn("not sure if the constant is correct! This changed with the commit with hash 68bbbd2")

                if self.debug:
                    print("Error Threshold:", error_threshold)

                omega_est: float = self._estimate_omega(elem, current_solution_set, error_threshold, delta)
                omega_estimates.append(omega_est)

                if elem in current_solution_set:
                    continue

                if omega_est > estimated_omega_bound:
                    # add this element to solution set and recompute omegas
                    current_solution_set.add(elem)
                    restart_omega_computations = True

                    if self.debug:
                        print("adding elem to solution set")
                    break

            if restart_omega_computations:
                restart_omega_computations = False
                continue

            elem_idx: int
            elem: E
            for elem_idx, elem in enumerate(current_solution_set):
                if omega_estimates[elem_idx] < - estimated_omega_bound:
                    current_solution_set.remove(elem)
                    restart_omega_computations = True

                    if self.debug:
                        print("removing elem from solution set")
                    break

            if restart_omega_computations:
                restart_omega_computations = False
                continue

            if self.debug:
                print("nb of elements added:", nb_of_adds)
                print("nb of elements removed:", nb_of_removes)

            if len(current_solution_set) == 0:
                print("the final set based on which is sampled is empty")
                warnings.warn("the final set based on which is sampled is empty")

            return sample_a_set_with_bias_delta_on_A(current_solution_set, self.ground_set, delta_prime)

    def _compute_OPT(self) -> float:
        """
        Estimate the optimal value for the objective function, by
         1. taking a sample subset of the possible elements, each element sampled with probability 1/2,
         2. evaluating the objective function for this random solution set
        """
        solution_set: Set[E] = self.rs_optimizer.optimize()

        return self.objective_function.evaluate(solution_set)

    def _estimate_omega(self, elem: E, solution_set: Set[E], error_threshold: float, delta: float) -> float:
        # NOTE: these lists are never emptied/re-initialized
        expected_values_of_function_include_x: List[float] = []
        expected_values_of_function_exclude_x: List[float] = []

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
                variance_exp_include / len(expected_values_of_function_include_x) + variance_exp_exclude / len(expected_values_of_function_exclude_x))

            if self.debug:
                print("Standard Error:", standard_error, ", Error Threshold:", error_threshold)

        return np.mean(expected_values_of_function_include_x) - np.mean(expected_values_of_function_exclude_x)

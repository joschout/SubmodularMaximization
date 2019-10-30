import numpy as np


class AbstractDoubleGreedySearchDecisionStrategy:
    def should_update_X(self, a: float, b: float, debug: bool) -> bool:
        raise NotImplementedError('abstract method')


class DeterministicDoubleGreedySearchDecisionStrategy(AbstractDoubleGreedySearchDecisionStrategy):
    def should_update_X(self, a: float, b: float, debug: bool) -> bool:

        should_update_X = a >= b

        if debug:
            print("\t\ta =", a)
            print("\t\tb =", b)

            if should_update_X:
                print("\ta >= b")
                print("\tUPDATE X_prev:")
            else:
                print("\ta < b")
                print("\tUPDATE Y_prev:")

        return should_update_X


class RandomizedDoubleGreedySearchDecisionStrategy(AbstractDoubleGreedySearchDecisionStrategy):
    def should_update_X(self, a: float, b: float, debug: bool) -> bool:
        a_prime: float = max(a, 0)
        b_prime: float = max(b, 0)

        prob_boundary: float
        if a_prime == 0 and b_prime == 0:
            prob_boundary = 1
        else:
            prob_boundary = a_prime / (a_prime + b_prime)

        random_value: float = np.random.uniform()

        if debug:
            print("\t\ta =", a, "-> a' = ", a_prime)
            print("\t\tb =", b, "-> b' = ", b_prime)
            print("\t\tprob_bound =", prob_boundary)
            print("\t\trand_val   = ", random_value)

        should_update_X = random_value <= prob_boundary

        if debug:
            if should_update_X:
                print("\trandom_value <= prob_boundary")
                print("\tUPDATE X_prev:")
            else:
                print("\trandom_value > prob_boundary")
                print("\tUPDATE Y_prev:")

        return should_update_X

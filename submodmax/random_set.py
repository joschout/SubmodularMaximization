import random
from typing import TypeVar, Set, Tuple

from .abstract_optimizer import AbstractOptimizer

E = TypeVar('E')


def sample_a_set_with_bias_delta_on_A(set_A: Set[E], ground_set: Set[E], delta: float) -> Set[E]:
    """
    Samples a subset of the ground set X with bias delta based on a set A,
    Fore e in A subseteq X:
        add e to the new subset with probability p = (1 + delta) / 2
    For e in X - A
        add e to the new subset with probability p' = (1 - delta) / 2

    Return both the newly sampled set, and which elements are added to A or removed from A to obtain that set.

    :param set_A:
    :param ground_set:
    :param delta:
    :return:
    """
    new_set_R_of_A_delta: Set[E] = set()

    p: float = (1 + delta) / 2
    q: float = 1 - p

    # sample in-set elements with prob. (delta + 1)/2
    elem: E
    for elem in set_A:
        random_val: float = random.uniform(a=0, b=1)
        if random_val <= p:
            new_set_R_of_A_delta.add(elem)

    # sample out-set elements with prob. (1 - delta)/2
    for elem in ground_set:
        if elem not in set_A:
            random_val: float = random.uniform(a=0, b=1)
            if random_val <= q:
                new_set_R_of_A_delta.add(elem)

    return new_set_R_of_A_delta


def sample_a_set_with_bias_delta_on_A_extended(set_A: Set[E], ground_set: Set[E],
                                               delta: float) -> Tuple[Set[E], Set[E], Set[E]]:
    """
    A set is sampled with bias delta based on a set A,
    IF elements in A
        are sampled independently with probability p = (1+ delta)/2
    AND
    IF elements in ground_set - A
        are sampled independently with probability q = 1 - p = (1-delta)/2

    Return both the newly sampled set, and which elements are added to A or removed from A to obtain that set.

    :param set_A:
    :param ground_set:
    :param delta:
    :return:
    """

    new_set_R_of_A_delta: Set[E] = set()
    added_elements: Set[E] = set()
    removed_elements: Set[E] = set()

    p: float = (1 + delta) / 2
    q: float = 1 - p

    elem: E
    for elem in set_A:
        random_val: float = random.uniform(a=0, b=1)
        if random_val <= p:
            new_set_R_of_A_delta.add(elem)
        else:
            removed_elements.add(elem)

    for elem in ground_set:
        if elem not in set_A:
            random_val: float = random.uniform()
            if random_val <= q:
                new_set_R_of_A_delta.add(elem)
            else:
                added_elements.add(elem)
    return new_set_R_of_A_delta, added_elements, removed_elements


class RandomSetOptimizer(AbstractOptimizer):
    """
    Random Set Algorithm:
        return R = X(1/2), a uniformly random subset of X.
    e.g Max-cut : choose a set of vertices S in a graph with vertices X,
        such that  S = max{ f(S): S in X}
            with f(S) = \\sum_{e\\in cut(S)} w(e)
                      = the sum of the weights of the crossing edges defined by the cut(S) in the graph.
        It is known that choosing a random subset is a good choice for Max Cut,

         achieving an approximation factor of 1/2.

    See also: Maximizing non-monotone submodular functions, Feige, Mirrokni, Vondrák

    Given:
        f: 2^X -> R+, or equivalently f: [0,1]^n -> R+ with |X|=n
        f is a submodular function
        OPT = max_{ S \\subset X } f(S)
        R = a uniformly randoms subset of X
    Then:
        E[ f(R) ] >= 1/4 * OPT
    """

    def __init__(self, ground_set: Set[E], probability: float = 0.5, debug: bool = True):
        super().__init__(objective_function=None, ground_set=ground_set, debug=debug)
        self.probability = probability

    def optimize(self) -> Set[E]:
        solution_set: Set[E] = set()

        elem: E
        for elem in self.ground_set:
            if random.uniform(a=0, b=1) <= self.probability:
                solution_set.add(elem)

        return solution_set

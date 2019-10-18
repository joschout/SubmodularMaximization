from typing import Set, TypeVar

E = TypeVar('E')


class AbstractObjectiveFunction:
    def evaluate(self, input_set: Set[E]) -> float:
        raise NotImplementedError('Abstract Method')


class AbstractOptimizer:
    def __init__(self, objective_function: AbstractObjectiveFunction, ground_set: Set[E], debug: bool = True):
        self.objective_function: AbstractObjectiveFunction = objective_function
        self.ground_set: Set[E] = ground_set
        self.debug: bool = debug

    def optimize(self) -> Set[E]:
        raise NotImplementedError("abstract method")

from typing import Set, TypeVar, Optional, Tuple

from .set_info import SetInfo

E = TypeVar('E')


class FuncInfo:

    def __init__(self, func_value: float):
        self.func_value: float = func_value


class AbstractSubmodularFunctionValueReuse:

    def evaluate(self, current_set_info: SetInfo,
                 previous_func_info: Optional[FuncInfo],
                 ) -> FuncInfo:
        raise NotImplementedError('abstract method')


class AbstractOptimizerValueReuse:

    def __init__(self, objective_function: AbstractSubmodularFunctionValueReuse, ground_set: Set[E],
                 debug: bool = True):
        self.objective_function: AbstractSubmodularFunctionValueReuse = objective_function
        self.ground_set: Set[E] = ground_set
        self.debug: bool = debug

    def optimize(self) -> Tuple[SetInfo, FuncInfo]:
        raise NotImplementedError("abstract method")

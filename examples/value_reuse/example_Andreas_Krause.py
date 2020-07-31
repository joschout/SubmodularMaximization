from typing import Set, TypeVar, Optional

from submodmax.value_reuse.abstract_optimizer import AbstractSubmodularFunctionValueReuse, AbstractOptimizerValueReuse, \
    FuncInfo
from submodmax.value_reuse.randomized_double_greedy_search import RandomizedDoubleGreedySearch
from submodmax.value_reuse.set_info import SetInfo

E = TypeVar('E')


class ExampleFuncInfo(FuncInfo):
    def __init__(self, func_value: float):
        super().__init__(func_value=func_value)

    @staticmethod
    def init_objective_function_value_info() -> 'ExampleFuncInfo':
        return ExampleFuncInfo(func_value=0)


class AndreasKrauseExampleObjectiveFunction(AbstractSubmodularFunctionValueReuse):
    def evaluate(self, current_set_info: SetInfo,
                 previous_func_info: Optional[ExampleFuncInfo],
                 ) -> ExampleFuncInfo:

        if previous_func_info is None:
            previous_func_info = ExampleFuncInfo.init_objective_function_value_info()

        # NOTE: the following is not necessary for the current example,
        # but illustrates how you can use information about the changes in the current set between evaluations.

        ground_set_size: int = current_set_info.get_ground_set_size()
        current_rule_set_size: int = current_set_info.get_current_set_size()

        # WARNING: during evaluation, this might not be updated with the added_elements!
        # Thus, you should not access this
        current_set: Set[E] = current_set_info.current_set

        # the elements added to the set from the previous evaluation:
        added_elements: Set[E] = current_set_info.get_added_elems()

        # the elements removed from the set in the previous evaluation:
        deleted_elements: Set[E] = current_set_info.get_deleted_elems()

        # the elements shared between the sets in the previous evaluation and the current evaluation
        intersection_previous_and_current_elements = current_set_info.get_intersection_previous_and_current_elems()

        current_set = intersection_previous_and_current_elements.union(added_elements)

        if current_set == set():
            return ExampleFuncInfo(func_value=0)
        elif current_set == {1}:
            return ExampleFuncInfo(func_value=-1)
        elif current_set == {2}:
            return ExampleFuncInfo(func_value=2)
        elif current_set == {1, 2}:
            return ExampleFuncInfo(func_value=0)
        else:
            raise Exception(f"The input set was not expected: {current_set_info.current_set}")


def run_example():
    """
    The example from the tutorial slides at www.submodularity.org
    Originally implemented by Andreas Krause (krausea@gmail.com) in his SFO toolbox in Matlab.

    The function:
        | input_set | output |
        |-----------|--------|
        | {}        | 0      |
        | {1}       | -1     |
        | {2}       | 2      |
        | {1, 2}    | 0      |

    The ground set: { 1, 2 }
    """

    ground_set: Set[int] = {1, 2}
    submodular_objective_function = AndreasKrauseExampleObjectiveFunction()

    optimizer: AbstractOptimizerValueReuse = RandomizedDoubleGreedySearch(
        objective_function=submodular_objective_function,
        ground_set=ground_set,
        debug=False
    )

    rule_set_info: SetInfo
    obj_func_val_info: ExampleFuncInfo
    rule_set_info, obj_func_val_info = optimizer.optimize()

    local_optimum: Set[int] = rule_set_info.current_set
    true_optimum: Set[int] = {2}
    print(local_optimum)
    if true_optimum == local_optimum:
        print(f"Found correct local optimum: {local_optimum}")
    else:
        print(f"Found {local_optimum}, but should be {true_optimum}")


if __name__ == '__main__':
    run_example()

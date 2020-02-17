from typing import Tuple

from submodmax.value_reuse.abstract_optimizer import AbstractOptimizerValueReuse
from value_reuse.abstract_optimizer import FuncInfo
from value_reuse.set_info import SetInfo


class GroundSetReturner(AbstractOptimizerValueReuse):
    def optimize(self) -> Tuple[SetInfo, FuncInfo]:
        ground_set_size = len(self.ground_set)
        empty_set = set()

        ground_set_info: SetInfo = SetInfo(
            ground_set_size=ground_set_size,
            current_set_size=ground_set_size,
            added_elems=self.ground_set,
            deleted_elems=empty_set,
            intersection_previous_and_current_elems=empty_set
        )
        ground_set_info.set_current_set(self.ground_set)

        func_info_ground_set = self.objective_function.evaluate(ground_set_info, previous_func_info=None)

        return ground_set_info, func_info_ground_set

from typing import Set

from abstract_optimizer import E
from .abstract_optimizer import AbstractOptimizer


class GroundSetReturner(AbstractOptimizer):
    def optimize(self) -> Set[E]:
        return self.ground_set

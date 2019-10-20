from typing import Set, TypeVar, Optional

E = TypeVar('E')


class AbstractSetInfo:
    def get_ground_set_size(self) -> int:
        raise NotImplementedError("abstract method")

    def get_current_set_size(self) -> int:
        raise NotImplementedError("abstract method")

    def get_added_elems(self):
        raise NotImplementedError("abstract method")

    def get_deleted_elems(self):
        raise NotImplementedError("abstract method")

    def get_intersection_previous_and_current_elems(self):
        raise NotImplementedError("abstract method")

    def set_current_set(self, current_set):
        raise NotImplementedError("abstract method")


class SetInfo(AbstractSetInfo):
    def get_ground_set_size(self) -> int:
        return self.ground_set_size

    def set_current_set(self, current_set: Set[E]) -> None:
        self.current_set = current_set

    def get_current_set_size(self) -> int:
        return self.current_set_size

    def get_added_elems(self) -> Set[E]:
        return self.added_elems

    def get_deleted_elems(self) -> Set[E]:
        return self.deleted_elems

    def get_intersection_previous_and_current_elems(self) -> Set[E]:
        return self.intersection_previous_and_current_elems

    def __init__(self, ground_set_size: int,
                 current_set_size: int,
                 added_elems: Optional[Set[E]],
                 deleted_elems: Optional[Set[E]],
                 intersection_previous_and_current_elems: Optional[Set[E]]
                 ):
        self.ground_set_size: int = ground_set_size

        self.current_set: Optional[Set[E]] = None
        self.current_set_size: int = current_set_size

        self.added_elems: Optional[Set[E]] = added_elems
        self.deleted_elems: Optional[Set[E]] = deleted_elems
        self.intersection_previous_and_current_elems: Optional[Set[E]] = intersection_previous_and_current_elems

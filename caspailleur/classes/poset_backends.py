from abc import ABC, abstractmethod
from collections.abc import Iterator
from functools import reduce
from typing import Self, Optional

POSET_BACKEND_REGISTRY: dict[str, type['PosetBackend']] = dict()

def register_poset_backend(key: str):
    def decorator(cls):
        POSET_BACKEND_REGISTRY[key] = cls
        return cls

    return decorator



class PosetBackend(ABC):
    def __init__(self, n_elements: Optional[int], leq_order: set[tuple[int, int]]) -> None:
        assert all(i <= j for i, j in leq_order)
        n_elements = n_elements if n_elements is not None else max(max(pair) for pair in leq_order)+1
        assert all((i, i) in leq_order for i in range(n_elements)) is not None

        self.leq_order = leq_order

    @property
    @abstractmethod
    def leq_order(self) -> set[tuple[int, int]]:
        raise NotImplementedError

    @leq_order.setter
    def leq_order(self, value: set[tuple[int, int]]) -> None:
        raise NotImplementedError

    def add(self, element: int, predecessors: set[int] = None, successors: set[int] = None) -> None:
        raise NotImplementedError

    def remove(self, element: int) -> None:
        raise NotImplementedError

    @property
    def n_elements(self) -> int:
        return max(max(pair) for pair in self.leq_order) + 1

    @property
    def elements(self) -> set[int]:
        return set(range(self.n_elements))

    def is_leq(self, element: int, other: int) -> bool:
        return (element, other) in self.leq_order

    def predecessors(self, element: int, reflexive_output: bool = True) -> set[int]:
        return {i for i in range(element+int(reflexive_output)) if self.is_leq(i, element)}

    def successors(self, element: int, reflexive_output: bool = True) -> set[int]:
        return {i for i in range(element+1-int(reflexive_output), self.n_elements) if self.is_leq(element, i)}

    def __copy__(self):
        return type(self)(self.n_elements, self.leq_order)

    def copy(self) -> Self:
        return self.__copy__()

    def __sub__(self, other: Self) -> Self:
        return type(self)(self.n_elements, self.leq_order - other.leq_order)

    def direct_predecessors(self, element: int) -> set[int]:
        predecessors_to_process = self.predecessors(element, reflexive_output=False)
        directs = set()
        while predecessors_to_process:
            closest_pred = max(predecessors_to_process)
            directs.add(closest_pred)
            predecessors_to_process.remove(closest_pred)
            predecessors_to_process -= self.predecessors(closest_pred)
        return directs

    def direct_successors(self, element: int) -> set[int]:
        successors_to_process = self.successors(element, reflexive_output=False)
        directs = set()
        while successors_to_process:
            closest_succ = min(successors_to_process)
            directs.add(closest_succ)
            successors_to_process.remove(closest_succ)
            successors_to_process -= self.successors(closest_succ)
        return directs

    def greatest_common_predecessor(self, *elements: int) -> Optional[int]:
        common_predecessors = reduce(set.intersection, (self.predecessors(el) for el in elements), self.elements)
        if not common_predecessors:
            return None
        gcp = max(common_predecessors)
        if common_predecessors == self.predecessors(gcp):
            return gcp
        return None

    def smallest_common_successor(self, *elements: int) -> Optional[int]:
        common_successors = reduce(set.intersection, (self.successors(el) for el in elements), self.elements)
        if not common_successors:
            return None
        scs = min(common_successors)
        if common_successors == self.successors(scs):
            return scs
        return None

    def min(self, *elements: int) -> Optional[int]:
        elements = self.elements if not elements else set(elements)
        min_cand = min(elements)
        if elements <= self.successors(min_cand):
            return min_cand
        return None

    def max(self, *elements: int) -> Optional[int]:
        elements = self.elements if not elements else set(elements)
        max_cand = max(elements)
        if elements <= self.predecessors(max_cand):
            return max_cand
        return None

    def __len__(self) -> int:
        return self.n_elements

    @property
    def direct_predecessors_pairs(self) -> Iterator[tuple[int, int]]:
        return ((i, j) for i in range(self.n_elements) for j in self.direct_predecessors(i))

    @property
    def direct_successors_pairs(self) -> Iterator[tuple[int, int]]:
        return ((i, j) for i in range(self.n_elements) for j in self.direct_successors(i))

    @property
    def predecessors_pairs(self) -> Iterator[tuple[int, int]]:
        return ((i, j) for i in range(self.n_elements) for j in self.predecessors(i))

    @property
    def successors_pairs(self) -> Iterator[tuple[int, int]]:
        return ((i, j) for i in range(self.n_elements) for j in self.successors(i))

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.n_elements))

    def __contains__(self, item: int | tuple[int, int]) -> bool:
        if isinstance(item, tuple) and len(item) == 2 and 0 <= item[0] < self.n_elements and 0 <= item[1] < self.n_elements:
            return self.is_leq(item[0], item[1])
        return 0 <= item < self.n_elements

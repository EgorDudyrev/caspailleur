from collections.abc import Hashable
from dataclasses import dataclass
from functools import reduce
from typing import TypeVar, Self, Optional

TElement = TypeVar('TElement', bound=Hashable)

@dataclass
class Poset:
    elements: set[TElement]
    leq_order: set[tuple[TElement, TElement]]

    def predecessors(self, element: TElement) -> set[TElement]:
        return {other for other in self.elements if (other, element) in self.leq_order and other!=element}

    def successors(self, element: TElement) -> set[TElement]:
        return {other for other in self.elements if (element, other) in self.leq_order and other!=element}

    def direct_predecessors(self, element: TElement) -> set[TElement]:
        predecessors_to_process = self.predecessors(element)
        directs = set()
        while predecessors_to_process:
            closest_pred = max(predecessors_to_process, key=map(len, self.predecessors))
            directs.add(closest_pred)
            predecessors_to_process.remove(closest_pred)
            predecessors_to_process -= self.predecessors(closest_pred)
        return directs

    def direct_successors(self, element: TElement) -> set[TElement]:
        successors_to_process = self.successors(element)
        directs = set()
        while successors_to_process:
            closest_succ = max(successors_to_process, key=map(len, self.successors))
            directs.add(closest_succ)
            successors_to_process.remove(closest_succ)
            successors_to_process -= self.successors(closest_succ)
        return directs

    @classmethod
    def from_direct_predecessors(cls, direct_predecessors: set[tuple[TElement, TElement]]) -> Self:
        elements = {elem for pair in direct_predecessors for elem in pair}
        leq_order = direct_predecessors
        while True:
            new_leq_order = set(leq_order)
            new_leq_order |= {(pair[0], other[1]) for pair in leq_order for other in leq_order if pair[1]==other[0]}
            if new_leq_order == leq_order:
                break
            leq_order = new_leq_order
        return cls(elements, leq_order)

    @classmethod
    def from_direct_successors(cls, direct_successors: set[tuple[TElement, TElement]]) -> Self:
        return cls.from_direct_predecessors({(pair[1], pair[0]) for pair in direct_successors})

    def greatest_common_predecessor(self, *elements: TElement) -> Optional[TElement]:
        common_predecessors = reduce(set.intersection, map(self.predecessors, elements), self.elements)
        gcp = max(common_predecessors, key=map(len, self.predecessors))
        if common_predecessors == self.predecessors(gcp) | {gcp}:
            return gcp
        return None

    def smallest_common_successor(self, *elements: TElement) -> Optional[TElement]:
        common_successors = reduce(set.intersection, map(self.successors, elements), self.elements)
        scs = max(common_successors, key=map(len, self.successors))
        if common_successors == self.successors(scs) | {scs}:
            return scs
        return None

    def supremum(self, *elements: TElement) -> Optional[TElement]:
        return self.smallest_common_successor(*elements)

    def infimum(self, *elements: TElement) -> Optional[TElement]:
        return self.greatest_common_predecessor(*elements)

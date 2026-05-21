from abc import ABC, abstractmethod
from collections.abc import Iterable
from functools import reduce
from typing import Literal, Union, Optional

from bitarray import bitarray
from bitarray.util import zeros as bazeros, ones as baones

from caspailleur.classes.formal_context import TAttribute
from caspailleur.algorithms.base_functions import powerset
from caspailleur.io import isets2bas

IMPLICATIONAL_REGISTRY: dict[str, type['ImplicationalSystemBackend']] = dict()


def register_implicational_backend(key: str):
    def decorator(cls):
        assert key not in IMPLICATIONAL_REGISTRY
        IMPLICATIONAL_REGISTRY[key] = cls
        return cls
    return decorator


class ImplicationalSystemBackend(ABC):
    @property
    @abstractmethod
    def implications(self) -> dict[frozenset[TAttribute], set[TAttribute]]:
        pass

    @abstractmethod
    def saturate(self, description: set[TAttribute]) -> set[TAttribute]:
        pass

    @abstractmethod
    def add(self, premise: Iterable[TAttribute], conclusion: Iterable[TAttribute]):
        ...

    @abstractmethod
    def remove(self, premise: Iterable[TAttribute], conclusion: Iterable[TAttribute]):
        pass

    def __call__(self, description: set[TAttribute]) -> set[TAttribute]:
        return self.saturate(description)

    def __contains__(self, item: set[TAttribute]) -> bool:
        return self.saturate(item) == item

    def __len__(self) -> int:
        return len(self.implications)

    def size(self) -> int:
        return sum(len(premise)+len(conclusion) for premise, conclusion in self.implications.items())

    @property
    def base_set(self) -> set[TAttribute]:
        return {attr for premise, conclusion in self.implications.items() for attr in premise | conclusion}

    def __iter__(self) -> Iterable[set[TAttribute]]:
        return (set(description) for description in powerset(self.base_set) if description in self)


@register_implicational_backend('Naive')
class NaiveImplicationalSystemBackend(ImplicationalSystemBackend):
    def __init__(self, implications: dict[frozenset[TAttribute], set[TAttribute]]):
        self._implications = dict(implications)

    @property
    def implications(self) -> dict[frozenset[TAttribute], set[TAttribute]]:
        return dict(self._implications)

    def saturate(self, description: set[TAttribute]) -> set[TAttribute]:
        closure = set(description)
        while True:
            closure_new = set(closure)
            for premise, conclusion in self._implications.items():
                if premise <= closure_new:
                    closure_new |= conclusion
            if closure == closure_new:
                break
            closure = closure_new
        return closure

    def add(self, premise: Iterable[TAttribute], conclusion: Iterable[TAttribute]):
        premise = frozenset(premise)
        if premise not in self._implications:
            self._implications[premise] = set()
        self._implications[premise] |= set(conclusion)

    def remove(self, premise: Iterable[TAttribute], conclusion: Iterable[TAttribute]):
        premise = frozenset(premise)
        if premise not in self._implications:
            return
        self._implications[premise] -= set(conclusion)


@register_implicational_backend('VerticalWild')
class VerticalWildImplicationalSystemBackend(ImplicationalSystemBackend):
    def __init__(self, implications: dict[frozenset[TAttribute], set[TAttribute]]):
        self._vertical_premises: list[bitarray] = list()  # for binary matrix Attribute X Premises-containing-it
        self._conclusions: list[bitarray] = list()  # for binary matrix Premise X Attributes-implied-by-it
        self._attribute_order: list[TAttribute] = list()  # mapping from index of _vertical_premises element to Attribute
        self._attribute_index_map: dict[TAttribute, int] = dict()  # mapping from an Attribute to its _vertical_premises columns
        for premise, conclusion in implications.items():
            self.add(premise, conclusion)

    @property
    def implications(self) -> dict[frozenset[TAttribute], set[TAttribute]]:
        premises = [list() for _ in range(len(self))]
        for attribute_idx, vertical_premises in enumerate(self._vertical_premises):
            attribute = self._attribute_order[attribute_idx]
            for premise_idx in vertical_premises.search(True):
                premises[premise_idx].append(attribute)

        premises = [frozenset(premise) for premise in premises]
        conclusions = [set(map(self._attribute_order.__getitem__, conclusion.search(True))) for conclusion in self._conclusions]
        return dict(zip(premises, conclusions))

    def saturate_ba(self, description_ba: bitarray) -> bitarray:
        empty_premise_list = bazeros(len(self))

        closure = bitarray(description_ba)
        already_covered_premises = bitarray(empty_premise_list)
        while True:
            covered_premises = ~reduce(bitarray.__or__, (self._vertical_premises[i] for i in closure.search(False)), empty_premise_list)
            conclusions_to_add = covered_premises & ~already_covered_premises
            if not conclusions_to_add.any():
                break

            closure = reduce(bitarray.__or__, (self._conclusions[i] for i in conclusions_to_add.search(True)), closure)
            already_covered_premises = covered_premises

        return closure

    def saturate(self, description: set[TAttribute]) -> set[TAttribute]:
        description_idxs = self._attributes2indices(description)
        description_ba = bitarray(next(isets2bas([description_idxs], len(self._attribute_order))))

        closure_ba = self.saturate_ba(description_ba)
        closure = {self._attribute_order[i] for i in closure_ba.search(True)}
        return closure

    def __len__(self) -> int:
        return len(self._conclusions)

    def _find_premise(self, premise_ba: bitarray) -> Optional[int]:
        index_options = baones(len(self))
        for attr_idx, is_present in enumerate(premise_ba):
            index_options &= self._vertical_premises[attr_idx] if is_present else ~self._vertical_premises[attr_idx]
        if index_options.count(True) != 1:
            return None
        return index_options.index(True)

    def _attributes2indices(self, iterable: Iterable[TAttribute]) -> list[int]:
        empty_premise_list = bazeros(len(self))

        itemset = []
        for attribute in iterable:
            if attribute not in self._attribute_index_map:
                self._attribute_index_map[attribute] = len(self._attribute_order)
                self._attribute_order.append(attribute)
                self._vertical_premises.append(bitarray(empty_premise_list))
                for conclusion in self._conclusions: conclusion.append(False)

            itemset.append(self._attribute_index_map[attribute])
        return itemset

    def add(self, premise: Iterable[TAttribute], conclusion: Iterable[TAttribute]):
        premise_idxs = self._attributes2indices(premise)
        conclusion_idxs = self._attributes2indices(conclusion)
        premise_ba, conclusion_ba = map(bitarray, isets2bas([premise_idxs, conclusion_idxs], len(self._attribute_order)))

        if (premise_idx := self._find_premise(premise_ba)) is not None:
            self._conclusions[premise_idx] |= conclusion_ba
            return

        for attribute_idx, is_activated in enumerate(premise_ba):
            self._vertical_premises[attribute_idx].append(is_activated)
        self._conclusions.append(conclusion_ba)


    def remove(self, premise: Iterable[TAttribute], conclusion: Iterable[TAttribute]):
        premise_idxs = self._attributes2indices(premise)
        conclusion_idxs = self._attributes2indices(conclusion)
        premise_ba, conclusion_ba = isets2bas([premise_idxs, conclusion_idxs], len(self._attribute_order))

        premise_idx = self._find_premise(premise_ba)
        if premise_idx is None:
            return
        self._conclusions[premise_idx] &= ~conclusion_ba
        if not self._conclusions[premise_idx].any():
            self._conclusions.pop(premise_idx)
            for premises in self._vertical_premises:
                premises.pop(premise_idx)

    def __iter__(self) -> Iterable[set[TAttribute]]:
        n = len(self._attribute_order)
        return ({self._attribute_order[i] for i in idxs} for idxs in powerset(range(n))
                if (ba := next(isets2bas([idxs], n))) == self.saturate_ba(ba))


class ImplicationalSystem:
    def __init__(self, implications: dict[frozenset[TAttribute], set[TAttribute]], backend_class: Union[type[ImplicationalSystemBackend], Literal[tuple(IMPLICATIONAL_REGISTRY)]] = 'Naive'):
        backend_class = IMPLICATIONAL_REGISTRY[backend_class] if not isinstance(backend_class, type) else backend_class
        assert issubclass(backend_class, ImplicationalSystemBackend)
        self._backend = backend_class(implications)

    @property
    def implications(self) -> dict[frozenset[TAttribute], set[TAttribute]]:
        return self.backend.implications

    @property
    def backend(self) -> ImplicationalSystemBackend:
        return self._backend

    def saturate(self, description: set[TAttribute]) -> set[TAttribute]:
        return self.backend.saturate(description)

    def __call__(self, description: set[TAttribute]) -> set[TAttribute]:
        return self.backend(description)

    def __contains__(self, item: set[TAttribute]) -> bool:
        return self.backend.__contains__(item)

    def add(self, premise: Iterable[TAttribute], conclusion: Iterable[TAttribute]):
        return self.backend.add(premise, conclusion)

    def __len__(self) -> int:
        return self.backend.__len__()

    def size(self) -> int:
        return self.backend.size()

    @property
    def base_set(self) -> set[TAttribute]:
        return self.backend.base_set

    def __iter__(self) -> Iterable[TAttribute]:
        return self.backend.__iter__()

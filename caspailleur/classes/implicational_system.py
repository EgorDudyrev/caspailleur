from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Literal, Union, Optional, overload

from bitarray import bitarray
from bitarray.util import zeros as bazeros, ones as baones, subset as basubset

from caspailleur.algorithms.base_functions import select_subsets_vertical_ba
from caspailleur.algorithms.implication_bases import saturate_vertical_ba
from caspailleur.algorithms.base_functions import powerset
from caspailleur.classes.formal_context import TAttribute

IMPLICATIONAL_REGISTRY: dict[str, type['ImplicationalSystemBackend']] = dict()


def register_implicational_backend(key: str):
    def decorator(cls):
        assert key not in IMPLICATIONAL_REGISTRY
        IMPLICATIONAL_REGISTRY[key] = cls
        return cls
    return decorator


class ImplicationalSystemBackend(ABC):
    def __init__(self, implications: dict[frozenset[TAttribute], set[TAttribute]]):
        self.implications = implications

    @property
    @abstractmethod
    def implications(self) -> dict[frozenset[TAttribute], set[TAttribute]]:
        pass

    @implications.setter
    def implications(self, value: dict[frozenset[TAttribute], set[TAttribute]]):
        if self.implications == value:
            return

        for premise, conclusion in self.implications.items():
            self.remove(premise, conclusion)
        for premise, conclusion in value.items():
            self.add(premise, conclusion)

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
        return all(conclusion <= item for premise, conclusion in self.implications.items() if premise <= item)

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
        self._implications = implications
        super().__init__(implications)
    
    @ImplicationalSystemBackend.implications.getter
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
        super().__init__(implications)

    @ImplicationalSystemBackend.implications.getter
    def implications(self) -> dict[frozenset[TAttribute], set[TAttribute]]:
        premises = [list() for _ in range(len(self))]
        for attribute_idx, vertical_premises in enumerate(self._vertical_premises):
            for premise_idx in vertical_premises.search(True):
                premises[premise_idx].append(attribute_idx)

        premises = [frozenset(self._idxs2attrs(attribute_idxs)) for attribute_idxs in premises]
        conclusions = [set(self._ba2attrs(conclusion_ba)) for conclusion_ba in self._conclusions]
        return dict(zip(premises, conclusions))

    def saturate_ba(self, description_ba: bitarray) -> bitarray:
        return saturate_vertical_ba(description_ba, self._vertical_premises, self._conclusions)

    def saturate(self, description: set[TAttribute]) -> set[TAttribute]:
        for attribute in description:
            self._add_attribute(attribute)

        description_ba = self._attrs2ba(description)
        closure_ba = self.saturate_ba(description_ba)
        closure = set(self._ba2attrs(closure_ba))
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

    def _add_attribute(self, attribute: TAttribute) -> None:
        if attribute in self._attribute_index_map:
            return None

        self._attribute_index_map[attribute] = len(self._attribute_order)
        self._attribute_order.append(attribute)
        self._vertical_premises.append(bazeros(len(self)))
        for conclusion in self._conclusions:
            conclusion.append(False)
        return None

    def add(self, premise: Iterable[TAttribute], conclusion: Iterable[TAttribute]):
        premise, conclusion = set(premise), set(conclusion)
        for attribute in premise | conclusion:
            self._add_attribute(attribute)

        premise_ba, conclusion_ba = self._attrs2ba(premise), self._attrs2ba(conclusion)
        premise_idx = self._find_premise(premise_ba)
        if premise_idx is not None:
            self._conclusions[premise_idx] |= conclusion_ba
            return

        for attribute_idx, is_activated in enumerate(premise_ba):
            self._vertical_premises[attribute_idx].append(is_activated)
        self._conclusions.append(conclusion_ba)


    def remove(self, premise: Iterable[TAttribute], conclusion: Iterable[TAttribute]):
        premise, conclusion = set(premise), set(conclusion)
        for attribute in premise | conclusion:
            self._add_attribute(attribute)

        premise_ba, conclusion_ba = self._attrs2ba(premise), self._attrs2ba(conclusion)
        premise_idx = self._find_premise(premise_ba)
        if premise_idx is None:
            return

        self._conclusions[premise_idx] &= ~conclusion_ba
        if not self._conclusions[premise_idx].any():
            self._conclusions.pop(premise_idx)
            for premises in self._vertical_premises:
                premises.pop(premise_idx)

    def __iter__(self) -> Iterable[set[TAttribute]]:
        descriptions_generator = map(set, powerset(self._attribute_order))
        models = filter(self.__contains__, descriptions_generator)
        return models

    def __contains__(self, item: set[TAttribute]) -> bool:
        premise_ba = self._attrs2ba(item)
        covered_premises = select_subsets_vertical_ba(premise_ba, self._vertical_premises)
        return all(basubset(self._conclusions[i], premise_ba) for i in covered_premises.search(True))

    def _ba2attrs(self, ba: bitarray) -> Iterable[TAttribute]:
        return (self._attribute_order[i] for i in ba.search(True))

    def _attrs2ba(self, attributes: set[TAttribute]) -> bitarray:
        ba = bazeros(len(self._attribute_order))
        for attr in attributes:
            ba[self._attribute_index_map[attr]] = True
        return ba

    def _idxs2attrs(self, indices: Iterable[int]) -> Iterable[TAttribute]:
        return (self._attribute_order[i] for i in indices)

    def _attrs2idxs(self, attributes: Iterable[TAttribute]) -> Iterable[int]:
        return (self._attribute_index_map[attr] for attr in attributes)

    def _idxs2ba(self, indices: Iterable[int]) -> bitarray:
        ba = bazeros(len(self._attribute_order))
        for idx in indices:
            ba[idx] = True
        return ba

    def _ba2idxs(self, ba: bitarray) -> Iterable[int]:
        return ba.search(True)

    @staticmethod
    def _ba_union_complete(bitarrays: Iterable[bitarray], initial: bitarray = None) -> bitarray:
        union = bitarray(initial) if initial is not None else None
        for ba in bitarrays:
            if union is None:
                union = ba
            union |= ba

        return union


class ImplicationalSystem:
    def __init__(self, implications: dict[frozenset[TAttribute], set[TAttribute]], backend_class: Union[type[ImplicationalSystemBackend], Literal[tuple(IMPLICATIONAL_REGISTRY)]] = 'Naive'):
        self.backend = backend_class
        self.implications = implications

    @property
    def implications(self) -> dict[frozenset[TAttribute], set[TAttribute]]:
        return self.backend.implications

    @implications.setter
    def implications(self, value: dict[frozenset[TAttribute], set[TAttribute]]) -> None:
        self.backend.implications = value

    @property
    def backend(self) -> ImplicationalSystemBackend:
        return self._backend

    @backend.setter
    @overload
    def backend(self, value: ImplicationalSystemBackend) -> None: ...

    @backend.setter
    @overload
    def backend(self, value: type[ImplicationalSystemBackend]) -> None: ...

    @backend.setter
    @overload
    def backend(self, value: Literal[tuple(IMPLICATIONAL_REGISTRY)]) -> None: ...

    @backend.setter
    def backend(self, value) -> None:
        existing_implications = self.implications if hasattr(self, '_backend') else dict()

        if isinstance(value, ImplicationalSystemBackend):
            new_backend = value.__class__(existing_implications) if existing_implications else value
        elif isinstance(value, type) and issubclass(value, ImplicationalSystemBackend):
            new_backend = value(existing_implications)
        elif isinstance(value, str) and value in IMPLICATIONAL_REGISTRY:
            new_backend = IMPLICATIONAL_REGISTRY[value](existing_implications)
        else:
            raise ValueError(f"Implicational system backend {value} is not supported.")

        self._backend: ImplicationalSystemBackend = new_backend


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

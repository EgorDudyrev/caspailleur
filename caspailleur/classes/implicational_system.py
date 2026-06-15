from collections.abc import Iterable
from typing import Literal, Union, overload, Callable

from caspailleur.classes.formal_context import TAttribute
from caspailleur.classes.implicational_backends import ImplicationalSystemBackend
from caspailleur.registries import IMPLICATIONAL_BACKEND_REGISTRY, CLOSURE_ITERATOR_REGISTRY


class ImplicationalSystem:
    def __init__(
            self,
            implications: dict[frozenset[TAttribute], set[TAttribute]],
            backend_class: Union[type[ImplicationalSystemBackend], Literal[tuple(IMPLICATIONAL_BACKEND_REGISTRY)]] = 'Naive',
            attributes_order: list[TAttribute] = None
    ):
        self._attributes_order = []
        self._attribute_index_map = dict()

        if attributes_order is not None:
            self.reorder_attributes(attributes_order)

        self.backend = backend_class
        self.implications = implications

    @property
    def implications(self) -> dict[frozenset[TAttribute], set[TAttribute]]:
        return {frozenset(self._idxs2attrs(premise)): set(self._idxs2attrs(conclusion))
                for premise, conclusion in self.backend.implications.items()}

    @implications.setter
    def implications(self, value: dict[frozenset[TAttribute], set[TAttribute]]) -> None:
        self.clear()
        for premise, conclusion in value.items():
            self.add(premise, conclusion)

    @property
    def unit_implications(self) -> list[tuple[frozenset[TAttribute], TAttribute]]:
        unit_implications = []
        for premise_idxs, unit_conclusion_idx in self.backend.unit_implications:
            premise = frozenset(self._idxs2attrs(premise_idxs))
            unit_conclusion = next(iter(self._idxs2attrs([unit_conclusion_idx])))
            unit_implications.append((premise, unit_conclusion))
        return unit_implications

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
    def backend(self, value: Literal[tuple(IMPLICATIONAL_BACKEND_REGISTRY)]) -> None: ...

    @backend.setter
    def backend(self, value) -> None:
        existing_implications = self.backend.implications if hasattr(self, '_backend') else dict()

        if isinstance(value, ImplicationalSystemBackend):
            new_backend = value.__class__(existing_implications) if existing_implications else value
        elif isinstance(value, type) and issubclass(value, ImplicationalSystemBackend):
            new_backend = value(existing_implications)
        elif isinstance(value, str) and value in IMPLICATIONAL_BACKEND_REGISTRY:
            new_backend = IMPLICATIONAL_BACKEND_REGISTRY[value](existing_implications)
        else:
            raise ValueError(f"Implicational system backend {value} is not supported.")

        self._backend: ImplicationalSystemBackend = new_backend

    def reorder_attributes(self, attributes_order: list[TAttribute]) -> None:
        assert set(attributes_order) == set(self._attributes_order)
        implications = dict(self.implications)

        self.clear()
        for attr in attributes_order:
            self._add_attribute(attr)
        for premise, conclusion in implications.items():
            self.add(premise, conclusion)

    def saturate(self, description: set[TAttribute], single_pass: bool = False) -> set[TAttribute]:
        return self._idxs2attrs(self.backend.saturate(self._attrs2idxs(description), single_pass=single_pass))

    def __call__(self, description: set[TAttribute]) -> set[TAttribute]:
        return self._idxs2attrs(self.backend(self._attrs2idxs(description)))

    def __contains__(self, item: set[TAttribute]) -> bool:
        return self.backend.__contains__(self._attrs2idxs(item))

    def add(self, premise: Iterable[TAttribute], conclusion: Iterable[TAttribute]):
        premise, conclusion = set(premise), set(conclusion)
        for attr in premise | conclusion:
            self._add_attribute(attr)

        return self.backend.add(self._attrs2idxs(premise), self._attrs2idxs(conclusion))

    def remove(self, premise: Iterable[TAttribute], conclusion: Iterable[TAttribute]):
        premise, conclusion = set(premise), set(conclusion)
        for attr in premise | conclusion:
            self._add_attribute(attr)

        self.backend.remove(self._attrs2idxs(premise), self._attrs2idxs(conclusion))

        legacy_attributes = premise | conclusion
        for p, c in self.implications.items():
            legacy_attributes -= p | c
        for attr in legacy_attributes:
            self._remove_attribute(attr)

    def clear(self) -> None:
        self.backend.clear()
        self._attributes_order = []
        self._attribute_index_map = dict()

    def __len__(self) -> int:
        return self.backend.__len__()

    def size(self) -> int:
        return self.backend.size()

    @property
    def base_set(self) -> set[TAttribute]:
        return set(self._attributes_order)

    def iterate_closures(
            self,
            algorithm: Literal[tuple(CLOSURE_ITERATOR_REGISTRY)] = 'CbO-FW',
            antimonotone_constraint_func: Callable[[Iterable[TAttribute]], bool] = None,
            single_saturation_pass: bool = False,
    ) -> Iterable[set[TAttribute]]:
        if antimonotone_constraint_func is not None:
            indexed_antimonotone_constraint_func = lambda idxs: antimonotone_constraint_func(self._idxs2attrs(idxs))
        else:
            indexed_antimonotone_constraint_func = None
        closure_iterator = self.backend.iterate_closures(algorithm, antimonotone_constraint_func=indexed_antimonotone_constraint_func, single_saturation_pass=single_saturation_pass)
        return map(self._idxs2attrs, closure_iterator)

    def count_closures(
            self,
            use_tqdm: bool = False,
            iteration_algorithm: Literal[tuple(CLOSURE_ITERATOR_REGISTRY)] = 'CbO-FW',
            antimonotone_constraint_func: Callable[[Iterable[TAttribute]], bool] = None,
            single_saturation_pass: bool = False,
    ) -> int:
        return self.backend.count_closures(use_tqdm, iteration_algorithm, antimonotone_constraint_func=antimonotone_constraint_func, single_saturation_pass=single_saturation_pass)

    def _idxs2attrs(self, indices: Iterable[int]) -> set[TAttribute]:
        return {self._attributes_order[idx] for idx in indices}

    def _attrs2idxs(self, attributes: Iterable[TAttribute]) -> set[int]:
        return {self._attribute_index_map[attr] for attr in attributes}

    def _add_attribute(self, attr: TAttribute) -> None:
        if attr in self._attribute_index_map:
            return

        idx = len(self._attributes_order)
        self._attributes_order.append(attr)
        self._attribute_index_map[attr] = idx
        self.backend.add_attribute()

    def _remove_attribute(self, attr: TAttribute) -> None:
        if attr not in self._attribute_index_map:
            return

        idx = self._attribute_index_map[attr]
        self._attributes_order.pop(idx)
        self._attribute_index_map.pop(attr)
        self.backend.remove_attribute(idx)

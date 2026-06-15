import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Literal, Optional, Callable, Iterable

from tqdm.auto import tqdm

from bitarray import bitarray
from bitarray.util import zeros as bazeros, ones as baones, subset as basubset

from caspailleur.registries import CLOSURE_ITERATOR_REGISTRY, register_implicational_backend
from caspailleur.algorithms.base_functions import select_subsets_vertical_ba
from caspailleur.algorithms.implication_bases import (
    saturate_vertical_ba,
    saturate_binary_ba
)
from caspailleur.classes.utils import filter_kwargs


class ImplicationalSystemBackend(ABC):
    def __init__(self, implications: dict[frozenset[int], set[int]]):
        self.implications = implications

    @property
    @abstractmethod
    def implications(self) -> dict[frozenset[int], set[int]]:
        pass

    @implications.setter
    def implications(self, value: dict[frozenset[int], set[int]]):
        if self.implications == value:
            return

        for premise, conclusion in self.implications.items():
            self.remove(premise, conclusion)

        n_attributes = max(max(premise | conclusion) for premise, conclusion in value.items())+1 if value else 0
        for _ in range(n_attributes):
            self.add_attribute()
        for premise, conclusion in value.items():
            self.add(premise, conclusion)

    @property
    def unit_implications(self) -> list[tuple[frozenset[int], int]]:
        return [(prem, unit_concl) for prem, conclusion in self.implications.items() for unit_concl in conclusion]

    @abstractmethod
    def saturate(self, description: Iterable[int], single_pass: bool = False) -> set[int]:
        pass

    @abstractmethod
    def add(self, premise: Iterable[int], conclusion: Iterable[int]):
        pass

    @abstractmethod
    def remove(self, premise: Iterable[int], conclusion: Iterable[int]):
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def add_attribute(self):
        pass

    @abstractmethod
    def remove_attribute(self, index: int):
        pass

    def __call__(self, description: set[int]) -> set[int]:
        return self.saturate(description)

    def __contains__(self, item: set[int]) -> bool:
        return all(conclusion <= item for premise, conclusion in self.implications.items() if premise <= item)

    def __len__(self) -> int:
        return len(self.implications)

    def size(self) -> int:
        return sum(len(premise) + len(conclusion) for premise, conclusion in self.implications.items())

    @property
    def base_set_len(self) -> int:
        return max(max(premise | conclusion) for premise, conclusion in self.implications.items()) + 1

    def iterate_closures(
            self,
            algorithm: Literal[tuple(CLOSURE_ITERATOR_REGISTRY)] = 'CbO-Forwardtrack',
            antimonotone_constraint_func: Callable[[Iterable[int]], bool] = None,
            single_saturation_pass: bool = False
    ) -> Iterable[set[int]]:
        if algorithm not in CLOSURE_ITERATOR_REGISTRY:
            raise ValueError(f'Algorithm {algorithm} is not supported as it is not found in CLOSURE_ITERATOR_REGISTRY')
        algo_func = CLOSURE_ITERATOR_REGISTRY[algorithm]

        kwargs_to_pass, defined_params, supported_params = filter_kwargs(self.iterate_closures, 1, locals(), set(), algo_func, 2)

        base_elements = set(range(self.base_set_len))
        saturate_func = partial(self.saturate, single_pass=single_saturation_pass)
        return algo_func(base_elements, saturate_func, **kwargs_to_pass)

    def count_closures(
            self,
            use_tqdm: bool = False,
            iteration_algorithm: Literal[tuple(CLOSURE_ITERATOR_REGISTRY)] = 'CbO-Forwardtrack',
            antimonotone_constraint_func: Callable[[Iterable[int]], bool] = None,
            single_saturation_pass: bool = False
    ) -> int:
        closures_iterator = self.iterate_closures(iteration_algorithm, antimonotone_constraint_func=antimonotone_constraint_func, single_saturation_pass=single_saturation_pass)
        return sum(1 for _ in tqdm(closures_iterator, desc='Count closures', disable=not use_tqdm, unit_scale=True))


@register_implicational_backend('Naive')
class NaiveImplicationalSystemBackend(ImplicationalSystemBackend):
    def __init__(self, implications: dict[frozenset[int], set[int]]):
        self._implications: dict[frozenset[int], set[int]] = dict()
        super().__init__(implications)

    @ImplicationalSystemBackend.implications.getter
    def implications(self) -> dict[frozenset[int], set[int]]:
        return dict(self._implications)

    def saturate(self, description: set[int], single_pass: bool = False) -> set[int]:
        closure = set(description)
        while True:
            closure_new = set(closure)
            for premise, conclusion in self._implications.items():
                if premise <= closure_new:
                    closure_new |= conclusion
            if closure == closure_new or single_pass:
                break
            closure = closure_new
        return closure

    def add(self, premise: Iterable[int], conclusion: Iterable[int]):
        premise = frozenset(premise)
        if premise not in self._implications:
            self._implications[premise] = set()
        self._implications[premise] |= set(conclusion)

    def remove(self, premise: Iterable[int], conclusion: Iterable[int]):
        premise = frozenset(premise)
        if premise not in self._implications:
            return
        self._implications[premise] -= set(conclusion)
        if len(self._implications[premise]) == 0:
            del self._implications[premise]

    def clear(self):
        self._implications.clear()

    def add_attribute(self):
        pass

    def remove_attribute(self, index: int):
        assert all(index not in premise | conclusion for premise, conclusion in self._implications.items())
        def pop_index(description):
            return (idx if idx < index else idx - 1  for idx in description if idx != index)

        self._implications = {frozenset(pop_index(premise)): set(pop_index(conclusion))
                              for premise, conclusion in self._implications.items()}



@register_implicational_backend('VerticalWild')
class VerticalWildImplicationalSystemBackend(ImplicationalSystemBackend):
    def __init__(self, implications: dict[frozenset[int], set[int]]):
        self._vertical_premises: list[bitarray] = []  # for binary matrix Attribute X Premises-containing-it
        self._conclusions: list[bitarray] = list()  # for binary matrix Premise X Attributes-implied-by-it
        super().__init__(implications)

    @ImplicationalSystemBackend.implications.getter
    def implications(self) -> dict[frozenset[int], set[int]]:
        premises = [list() for _ in range(len(self))]
        for attribute_idx, vertical_premises in enumerate(self._vertical_premises):
            for premise_idx in vertical_premises.search(True):
                premises[premise_idx].append(attribute_idx)

        premises = [frozenset(attribute_idxs) for attribute_idxs in premises]
        conclusions = [set(self._ba2idxs(conclusion_ba)) for conclusion_ba in self._conclusions]
        return dict(zip(premises, conclusions))

    def saturate_ba(self, description_ba: bitarray, single_pass: bool = False) -> bitarray:
        return saturate_vertical_ba(description_ba, self._vertical_premises, self._conclusions, single_pass=single_pass)

    def saturate(self, description: Iterable[int], single_pass: bool = False) -> set[int]:
        return set(self._ba2idxs(self.saturate_ba(self._idxs2ba(description), single_pass=single_pass)))

    def __len__(self) -> int:
        return len(self._conclusions)

    def size(self) -> int:
        premise_size = sum(vertical_premise.count() for vertical_premise in self._vertical_premises)
        conclusion_size = sum(conclusion.count() for conclusion in self._conclusions)
        return premise_size + conclusion_size

    def _find_premise(self, premise_ba: bitarray) -> Optional[int]:
        index_options = baones(len(self))
        for attr_idx, is_present in enumerate(premise_ba):
            index_options &= self._vertical_premises[attr_idx] if is_present else ~self._vertical_premises[attr_idx]
        if index_options.count(True) != 1:
            return None
        return index_options.index(True)

    def add_attribute(self) -> None:
        self._vertical_premises.append(bazeros(len(self)))
        for conclusion in self._conclusions:
            conclusion.append(False)

    def remove_attribute(self, index: int) -> None:
        self._vertical_premises.pop(index)
        for conclusion in self._conclusions:
            conclusion.pop(index)

    def add(self, premise: Iterable[int], conclusion: Iterable[int]):
        premise, conclusion = list(premise), list(conclusion)
        assert max(premise) < len(self._vertical_premises), f"The system only supports {len(self._vertical_premises)} attributes, but is given attribute #{max(premise)}"
        assert max(conclusion) < len(self._vertical_premises), f"The system only supports {len(self._vertical_premises)} attributes, but is given attribute #{max(conclusion)}"

        premise_ba, conclusion_ba = self._idxs2ba(premise), self._idxs2ba(conclusion)
        premise_idx = self._find_premise(premise_ba)
        if premise_idx is not None:
            self._conclusions[premise_idx] |= conclusion_ba
            return

        for attribute_idx, is_activated in enumerate(premise_ba):
            self._vertical_premises[attribute_idx].append(is_activated)
        self._conclusions.append(conclusion_ba)

    def remove(self, premise: Iterable[int], conclusion: Iterable[int]):
        premise_ba, conclusion_ba = self._idxs2ba(premise), self._idxs2ba(conclusion)
        premise_idx = self._find_premise(premise_ba)
        if premise_idx is None:
            return

        self._conclusions[premise_idx] &= ~conclusion_ba
        if not self._conclusions[premise_idx].any():
            self._conclusions.pop(premise_idx)
            for premises in self._vertical_premises:
                premises.pop(premise_idx)

    def clear(self):
        self._vertical_premises.clear()
        self._conclusions.clear()

    def __contains__(self, item: set[int]) -> bool:
        premise_ba = self._idxs2ba(item)
        covered_premises = select_subsets_vertical_ba(premise_ba, self._vertical_premises)
        return all(basubset(self._conclusions[i], premise_ba) for i in covered_premises.search(True))

    def _idxs2ba(self, indices: Iterable[int]) -> bitarray:
        ba = bazeros(len(self._vertical_premises))
        for idx in indices:
            ba[idx] = True
        return ba

    def _ba2idxs(self, ba: bitarray) -> Iterable[int]:
        return ba.search(True)


@register_implicational_backend('BitBinary')
class BitBinaryImplicationalSystemBackend(ImplicationalSystemBackend):
    def __init__(self, implications: dict[frozenset[int], set[int]]):
        self._conclusions_per_attribute: list[bitarray] = list()
        super().__init__(implications)

    @ImplicationalSystemBackend.implications.getter
    def implications(self) -> dict[frozenset[int], set[int]]:
        return {frozenset({attr}): set(self._ba2idxs(conclusion_ba))
                for attr, conclusion_ba in enumerate(self._conclusions_per_attribute)}

    def saturate_ba(self, description_ba: bitarray, single_pass: bool = False) -> bitarray:
        return saturate_binary_ba(description_ba, self._conclusions_per_attribute, single_pass=single_pass)

    def saturate(self, description: Iterable[int], single_pass: bool = False) -> set[int]:
        return set(self._ba2idxs(self.saturate_ba(self._idxs2ba(description), single_pass=single_pass)))

    def __len__(self) -> int:
        return len(self._conclusions_per_attribute)

    def size(self) -> int:
        premise_size = len(self._conclusions_per_attribute)
        conclusion_size = sum(conclusion.count() for conclusion in self._conclusions_per_attribute)
        return premise_size + conclusion_size

    def _find_premise(self, premise_ba: bitarray) -> Optional[int]:
        if premise_ba.count() != 1:
            return None
        return premise_ba.index(True)

    def add_attribute(self) -> None:
        self._conclusions_per_attribute.append(bazeros(len(self)))
        for conclusion in self._conclusions_per_attribute:
            conclusion.append(False)

    def remove_attribute(self, index: int) -> None:
        self._conclusions_per_attribute.pop(index)
        for conclusion in self._conclusions_per_attribute:
            conclusion.pop(index)

    def add(self, premise: Iterable[int], conclusion: Iterable[int]):
        premise, conclusion = list(premise), list(conclusion)
        n_attributes = len(self._conclusions_per_attribute)
        assert max(premise) < n_attributes, f"The system only supports {n_attributes} attributes, but is given attribute #{max(premise)}"
        assert max(conclusion) < n_attributes, f"The system only supports {n_attributes} attributes, but is given attribute #{max(conclusion)}"

        if len(premise) != 1:
            warnings.warn(f'This implicational backend ({self.__class__.__name__}) can also support binary implications, '
                          f'that are implications with exactly one element in the premise. '
                          f'Given premise {premise} contains more/less than one element and so will be omitted.')
            return

        attribute_idx = premise[0]
        self._conclusions_per_attribute[attribute_idx] |= self._idxs2ba(conclusion)

    def remove(self, premise: Iterable[int], conclusion: Iterable[int]):
        premise, conclusion = list(premise), list(conclusion)
        if len(premise) != 1:
            return

        premise_idx = premise[0]
        if premise_idx >= len(self._conclusions_per_attribute):
            return

        conclusion_ba = self._idxs2ba(conclusion)

        self._conclusions_per_attribute[premise_idx] &= ~conclusion_ba

    def clear(self):
        self._conclusions_per_attribute.clear()

    def _idxs2ba(self, indices: Iterable[int]) -> bitarray:
        ba = bazeros(len(self._conclusions_per_attribute))
        for idx in indices:
            ba[idx] = True
        return ba

    def _ba2idxs(self, ba: bitarray) -> Iterable[int]:
        return ba.search(True)

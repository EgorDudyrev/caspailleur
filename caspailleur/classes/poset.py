import operator
from collections.abc import Hashable, Callable, Iterator, Iterable
from functools import partial
from typing import TypeVar, Self, Optional, Literal

import matplotlib.pyplot as plt
import networkx as nx

from caspailleur.algorithms.layouts import LINE_LAYOUT_REGISTRY
from caspailleur.classes.utils import filter_kwargs
from caspailleur.classes.poset_backends import PosetBackend, POSET_BACKEND_REGISTRY

TElement = TypeVar('TElement', bound=Hashable)


class Poset:
    def __init__(
            self,
            elements: set[TElement], leq_order: set[tuple[TElement, TElement]],
            backend: Literal[tuple[POSET_BACKEND_REGISTRY]] = 'Naive'
    ):
        self.backend = backend
        self._elements, self._element_index_map = [], dict()

        self.leq_order = set(map(tuple, leq_order)) | {(el, el) for el in elements}
        self._init_measures()

    @property
    def backend(self) -> PosetBackend:
        return self._backend

    @backend.setter
    def backend(self, value):
        existing_leq = self.backend.leq_order if hasattr(self, '_backend') else set()

        if isinstance(value, PosetBackend):
            new_backend = value.__class__(existing_leq) if existing_leq else value
        elif isinstance(value, type) and issubclass(value, PosetBackend):
            new_backend = value(existing_leq)
        elif isinstance(value, str) and value in POSET_BACKEND_REGISTRY:
            new_backend = POSET_BACKEND_REGISTRY[value](existing_leq)
        else:
            raise ValueError(f"Poset backend {value} is not supported.")

        self._backend: PosetBackend = new_backend

    @property
    def elements(self) -> set[TElement]:
        return {self._elements[i] for i in self.backend.elements}

    @property
    def leq_order(self) -> set[tuple[TElement, TElement]]:
        return {(self._elements[i], self._elements[j]) for i, j in self.backend.leq_order}

    @leq_order.setter
    def leq_order(self, value: set[tuple[TElement, TElement]]) -> None:
        self.clear()
        predecessors = dict()
        for a, b in value:
            if b not in predecessors: predecessors[b] = set()
            predecessors[b].add(a)

        for b in sorted(predecessors, key=lambda el: len(predecessors[el])):
            self.add(b, predecessors[b])

    def predecessors(self, element: TElement, reflexive_output: bool = True) -> set[TElement]:
        return self._idxs2elements(self.backend.predecessors(self._element_index_map[element], reflexive_output))

    def successors(self, element: TElement, reflexive_output: bool = True) -> set[TElement]:
        return self._idxs2elements(self.backend.successors(self._element_index_map[element], reflexive_output))

    def direct_predecessors(self, element: TElement) -> set[TElement]:
        return self._idxs2elements(self.backend.direct_predecessors(self._element_index_map[element]))

    def direct_successors(self, element: TElement) -> set[TElement]:
        return self._idxs2elements(self.backend.direct_successors(self._element_index_map[element]))

    @property
    def direct_predecessors_pairs(self) -> Iterator[tuple[TElement, TElement]]:
        return ((self._elements[i], self._elements[j]) for i, j in self.backend.direct_predecessors_pairs)

    @property
    def direct_successors_pairs(self) -> Iterator[tuple[TElement, TElement]]:
        return ((self._elements[i], self._elements[j]) for i, j in self.backend.direct_successors_pairs)

    @property
    def predecessors_pairs(self) -> Iterator[tuple[TElement, TElement]]:
        return ((self._elements[i], self._elements[j]) for i, j in self.backend.predecessors_pairs)

    @property
    def successors_pairs(self) -> Iterator[tuple[TElement, TElement]]:
        return ((self._elements[i], self._elements[j]) for i, j in self.backend.successors_pairs)

    def __iter__(self) -> Iterator[TElement]:
        return (self._elements[i] for i in self.backend.elements)

    def __contains__(self, item: TElement | tuple[TElement, TElement]) -> bool:
        if isinstance(item, tuple) and len(item) == 2 and item[0] in self._element_index_map and item[1] in self._element_index_map:
            return self.backend.__contains__((self._element_index_map[item[0]], self._element_index_map[item[1]]))
        return item in self._element_index_map

    def __len__(self) -> int:
        return self.backend.n_elements

    def add(self, element: TElement, predecessors: set[TElement] = None, successors: set[TElement] = None) -> None:
        predecessors = predecessors - {element} if predecessors is not None else set()
        for predecessor in list(predecessors):
            predecessors |= self.predecessors(predecessor)

        successors = successors - {element} if successors is not None else set()
        for successor in list(successors):
            successors |= self.successors(successor)

        if element not in self._element_index_map:
            element_idx = min((self._element_index_map[el] for el in successors), default=len(self._element_index_map))
            print(f'new element idx: {element=} {element_idx=}')
            self._elements.insert(element_idx, element)
            self._element_index_map = {el: i + int(i >= element_idx) for el, i in self._element_index_map.items()}
            self._element_index_map[element] = element_idx
            self.backend.add_element(element_idx)

        print('add', element, predecessors)
        print('add idx', self._element_index_map[element], self._elements2idxs(predecessors))
        self.backend.add(self._element_index_map[element], self._elements2idxs(predecessors), self._elements2idxs(successors))

    def remove(self, element: TElement) -> None:
        element_idx = self._element_index_map[element]
        self.backend.remove(element_idx)
        self._elements.remove(element_idx)
        del self._element_index_map[element]
        self._element_index_map = {el: i - int(i >= element_idx) for el, i in self._element_index_map.items()}

    def __copy__(self) -> Self:
        return type(self)(self.elements, self.leq_order)

    def copy(self) -> Self:
        return self.__copy__()

    def __sub__(self, other: TElement) -> Self:
        backend_diff = self.backend - other.backend
        leq_diff = {(self._elements[i], self._elements[j]) for i, j in backend_diff.leq_order}
        elements = {el for pair in leq_diff for el in pair}
        return type(self)(elements, leq_diff)


    @classmethod
    def from_direct_predecessors(cls, direct_predecessors: set[tuple[TElement, TElement]]) -> Self:
        elements = {elem for pair in direct_predecessors for elem in pair}
        all_predecessors = {el: set() for el in elements}
        for el, pred in direct_predecessors:
            all_predecessors[el].add(pred)

        new_added = True
        while new_added:
            new_added = False
            for el in all_predecessors:
                for pred in list(all_predecessors[el]):
                    if not all_predecessors[pred] <= all_predecessors[el]:
                        new_added = True
                        all_predecessors[el] |= all_predecessors[pred]

        leq_order = {(el, pred) for el, preds in all_predecessors.items() for pred in preds}
        return cls(elements, leq_order)

    @classmethod
    def from_direct_successors(cls, direct_successors: set[tuple[TElement, TElement]]) -> Self:
        return cls.from_direct_predecessors({(pair[1], pair[0]) for pair in direct_successors})

    @classmethod
    def from_functional_order(cls, elements: set[TElement], leq_func: Callable[[TElement, TElement], bool] = operator.le) -> Self:
        return cls(elements, {(a, b) for a in elements for b in elements if leq_func(a, b)})

    def greatest_common_predecessor(self, *elements: TElement) -> Optional[TElement]:
        gcp = self.backend.greatest_common_predecessor(*(self._element_index_map[el] for el in elements))
        return self._elements[gcp] if gcp is not None else None

    def smallest_common_successor(self, *elements: TElement) -> Optional[TElement]:
        scs = self.backend.smallest_common_successor(*self._elements2idxs(elements))
        return self._elements[scs] if scs is not None else None

    def supremum(self, *elements: TElement) -> Optional[TElement]:
        return self.smallest_common_successor(*self._elements2idxs(elements))

    def infimum(self, *elements: TElement) -> Optional[TElement]:
        return self.greatest_common_predecessor(*self._elements2idxs(elements))

    def min(self, *elements: TElement) -> Optional[TElement]:
        min_ = self.backend.min(*self._elements2idxs(elements))
        return self._element_index_map[min_] if min_ is not None else None


    def max(self, *elements: TElement) -> Optional[TElement]:
        max_ = self.backend.max(*self._elements2idxs(elements))
        return self._element_index_map[max_] if max_ is not None else None

    def to_networkx(self, arrow_direction: Literal['ascending', 'descending'] = 'ascending') -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_nodes_from(self.elements)
        neighbours_func = self.direct_successors if arrow_direction == 'ascending' else self.direct_predecessors
        graph.add_edges_from({(node, neighbour) for node in self.elements for neighbour in neighbours_func(node)})
        return graph

    def line_layout(
            self, layout_type: Literal[tuple(LINE_LAYOUT_REGISTRY)] = 'odis-sugiyama', smallest_on_top: bool = False,
            y_position: dict[TElement, float] = None
    ) -> dict[TElement, tuple[float, float]]:
        assert layout_type in LINE_LAYOUT_REGISTRY, f'Unsupported layout type {layout_type}'
        layout_func = LINE_LAYOUT_REGISTRY[layout_type]
        kwargs_to_pass, defined_kwargs, supported_kwargs = filter_kwargs(self.line_layout, 2, locals(), set(), layout_func, 2)

        # Set up some default values
        undefined_kwargs = supported_kwargs - defined_kwargs
        if 'start_' in undefined_kwargs:
            kwargs_to_pass['start_'] = self.min()
        if 'y_position' in undefined_kwargs:
            kwargs_to_pass['y_position'] = {el: len(self.predecessors(el)) for el in self.elements}

        pos = layout_func(self.elements, set(self.direct_successors_pairs), **kwargs_to_pass)

        if smallest_on_top:
            max_y = max(y for _, (_, y) in pos.items())
            pos = {elem: (x, max_y - y) for elem, (x, y) in pos.items()}
        return {elem: (float(x), float(y)) for elem, (x, y) in pos.items()}

    def draw(self, ax: plt.Axes = None, layout_type: Literal[tuple(LINE_LAYOUT_REGISTRY)] = 'odis-sugiyama', **draw_kwargs) -> None:
        ax = plt.gca() if ax is None else ax
        graph = self.to_networkx()
        pos = self.line_layout(layout_type=layout_type, smallest_on_top=False)
        nx.draw(graph, pos=pos, ax=ax, with_labels=True, **draw_kwargs)

    @property
    def measures(self):# -> dict[tuple(POSET_MEASURE_REGISTRY), PosetMeasureProtocol]:
        return self._measures

    def _init_measures(self):
        from caspailleur.classes.poset_measures import POSET_MEASURE_REGISTRY, PosetMeasureProtocol
        class MeasuresRegistry:
            def __init__(registry_self):
                super().__init__()
                for func_name, func in POSET_MEASURE_REGISTRY.items():
                    partial_func = partial(func, poset=self)
                    setattr(registry_self, func_name, partial_func)


        self._measures = MeasuresRegistry()

    @property
    def T(self):
        return self.__class__(self.elements, {pair[::-1] for pair in self.leq_order})

    def clear(self) -> None:
        self.backend.clear()
        self._elements = []
        self._element_index_map = dict()

    def _elements2idxs(self, elements: Iterable[TElement]) -> set[int]:
        return {self._element_index_map[el] for el in elements}

    def _idxs2elements(self, indices: Iterable[int]) -> set[TElement]:
        return {self._elements[i] for i in indices}

    def downset(self, element: TElement, reflexive_relation: bool = True) -> set[TElement]:
        return self.predecessors(element, reflexive_relation)

    def upset(self, element: TElement, reflexive_relation: bool = True) -> set[TElement]:
        return self.successors(element, reflexive_relation)

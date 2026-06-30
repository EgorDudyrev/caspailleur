import operator
from collections.abc import Hashable, Callable, Iterator
from functools import reduce, partial
from typing import TypeVar, Self, Optional, Literal

import matplotlib.pyplot as plt
import networkx as nx

from caspailleur.algorithms.layouts import LINE_LAYOUT_REGISTRY
from caspailleur.classes.utils import filter_kwargs

TElement = TypeVar('TElement', bound=Hashable)


class Poset:
    def __init__(self, elements: set[TElement], leq_order: set[tuple[TElement, TElement]]):
        self.elements = set(elements)
        self.leq_order = set(map(tuple, leq_order)) | {(el, el) for el in self.elements}
        self._init_measures()

    def predecessors(self, element: TElement, reflexive_output: bool = True) -> set[TElement]:
        return {other for other in self.elements if (other, element) in self.leq_order and (reflexive_output or other != element)}

    def successors(self, element: TElement, reflexive_output: bool = True) -> set[TElement]:
        return {other for other in self.elements if (element, other) in self.leq_order and (reflexive_output or other != element)}

    def direct_predecessors(self, element: TElement) -> set[TElement]:
        predecessors_to_process = self.predecessors(element, reflexive_output=False)
        directs = set()
        while predecessors_to_process:
            closest_pred = max(predecessors_to_process, key=lambda pred: len(self.predecessors(pred)))
            directs.add(closest_pred)
            predecessors_to_process.remove(closest_pred)
            predecessors_to_process -= self.predecessors(closest_pred)
        return directs

    def direct_successors(self, element: TElement) -> set[TElement]:
        successors_to_process = self.successors(element, reflexive_output=False)
        directs = set()
        while successors_to_process:
            closest_succ = max(successors_to_process, key=lambda succ: len(self.successors(succ)))
            directs.add(closest_succ)
            successors_to_process.remove(closest_succ)
            successors_to_process -= self.successors(closest_succ)
        return directs

    @property
    def direct_predecessors_pairs(self) -> Iterator[tuple[TElement, TElement]]:
        return ((el, pred) for el in self.elements for pred in self.direct_predecessors(el))

    @property
    def direct_successors_pairs(self) -> Iterator[tuple[TElement, TElement]]:
        return ((el, suc) for el in self.elements for suc in self.direct_successors(el))

    @property
    def predecessors_pairs(self) -> Iterator[tuple[TElement, TElement]]:
        return ((el, pred) for el in self.elements for pred in self.predecessors(el))

    @property
    def successors_pairs(self) -> Iterator[tuple[TElement, TElement]]:
        return ((el, suc) for el in self.elements for suc in self.successors(el))

    def __iter__(self) -> Iterator[TElement]:
        return iter(sorted(self.elements, key=lambda el: len(self.successors(el))))

    def __contains__(self, item: TElement | tuple[TElement, TElement]) -> bool:
        if isinstance(item, tuple) and len(item) == 2 and item[0] in self.elements and item[1] in self.elements:
            return item in self.successors_pairs
        return item in self.elements

    def __len__(self) -> int:
        return len(self.elements)

    def add(self, element: TElement, predecessors: set[TElement] = None, successors: set[TElement] = None) -> None:
        predecessors = predecessors if predecessors is not None else set()
        for predecessor in list(predecessors):
            predecessors |= self.predecessors(predecessor)

        successors = successors if successors is not None else set()
        for successor in list(successors):
            successors |= self.successors(successor)

        self.elements.add(element)
        self.leq_order |= {(predecessor, element) for predecessor in predecessors}
        self.leq_order |= {(element, successor) for successor in successors}

    def remove(self, element: TElement) -> None:
        self.leq_order = {(a, b) for a, b in self.leq_order if element != a and element != b}
        self.elements.remove(element)

    def __copy__(self) -> Self:
        return type(self)(self.elements, self.leq_order)

    def copy(self) -> Self:
        return self.__copy__()

    def __sub__(self, other: TElement) -> Self:
        new_poset = self.copy()
        for element in other:
            new_poset.remove(element)
        return new_poset

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
        common_predecessors = reduce(set.intersection, (self.predecessors(el) for el in elements), self.elements)
        if not common_predecessors:
            return None
        gcp = max(common_predecessors, key=lambda pred: len(self.predecessors(pred)))
        if common_predecessors == self.predecessors(gcp):
            return gcp
        return None

    def smallest_common_successor(self, *elements: TElement) -> Optional[TElement]:
        common_successors = reduce(set.intersection, (self.successors(el) for el in elements), self.elements)
        if not common_successors:
            return None
        scs = max(common_successors, key=lambda suc: len(self.successors(suc)))
        if common_successors == self.successors(scs):
            return scs
        return None

    def supremum(self, *elements: TElement) -> Optional[TElement]:
        return self.smallest_common_successor(*elements)

    def infimum(self, *elements: TElement) -> Optional[TElement]:
        return self.greatest_common_predecessor(*elements)

    def min(self, *elements: TElement) -> Optional[TElement]:
        elements = self.elements if not elements else set(elements)
        min_cand = min(elements, key=lambda el: len(self.predecessors(el)))
        if elements <= self.successors(min_cand):
            return min_cand
        return None

    def max(self, *elements: TElement) -> Optional[TElement]:
        elements = self.elements if not elements else set(elements)
        max_cand = max(elements, key=lambda el: len(self.predecessors(el)))
        if elements <= self.predecessors(max_cand):
            return max_cand
        return None

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

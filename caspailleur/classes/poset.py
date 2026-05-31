import operator
from collections.abc import Hashable, Callable, Iterator
from functools import reduce
from typing import TypeVar, Self, Optional, Literal

import matplotlib.pyplot as plt
import networkx as nx

from caspailleur.registries import LINE_LAYOUT_REGISTRY
from caspailleur.classes.utils import filter_kwargs

TElement = TypeVar('TElement', bound=Hashable)


class Poset:
    def __init__(self, elements: set[TElement], leq_order: set[TElement]):
        self.elements = set(elements)
        self.leq_order = set(map(tuple, leq_order))

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

    @classmethod
    def from_functional_order(cls, elements: set[TElement], leq_func: Callable[[TElement, TElement], bool] = operator.le) -> Self:
        return cls(elements, {(a, b) for a in elements for b in elements if leq_func(a, b)})

    def greatest_common_predecessor(self, *elements: TElement) -> Optional[TElement]:
        common_predecessors = reduce(set.intersection, (self.predecessors(el) for el in elements), self.elements)
        gcp = max(common_predecessors, key=lambda pred: len(self.predecessors(pred)))
        if common_predecessors == self.predecessors(gcp):
            return gcp
        return None

    def smallest_common_successor(self, *elements: TElement) -> Optional[TElement]:
        common_successors = reduce(set.intersection, (self.successors(el) for el in elements), self.elements)
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
            self, layout_type: Literal[tuple(LINE_LAYOUT_REGISTRY)], smallest_on_top: bool = False,
            y_position: dict[TElement, float] = None
    ) -> dict[TElement, tuple[float, float]]:
        assert layout_type in LINE_LAYOUT_REGISTRY, f'Unsupported layout type {layout_type}'
        layout_func = LINE_LAYOUT_REGISTRY[layout_type]
        kwargs_to_pass, defined_kwargs, supported_kwargs = filter_kwargs(self.line_layout, 3, locals(), set(), layout_func, 2)

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

    def draw(self, ax: plt.Axes = None, layout_type: Literal[tuple(LINE_LAYOUT_REGISTRY)] = 'nx-BFS', **draw_kwargs) -> None:
        ax = plt.gca() if ax is None else ax
        graph = self.to_networkx()
        pos = self.line_layout(layout_type=layout_type, smallest_on_top=False)
        nx.draw(graph, pos=pos, ax=ax, with_labels=True, **draw_kwargs)

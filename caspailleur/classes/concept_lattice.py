from caspailleur.classes.poset import Poset
from caspailleur.classes.formal_context import FormalContext
from caspailleur.registries import CLOSURE_ITERATOR_REGISTRY

from tqdm.notebook import tqdm
from typing import NamedTuple

Concept = NamedTuple('FormalConcept', [('extent', frozenset), ('intent', frozenset)])


class ConceptLattice(Poset):
    @classmethod
    def from_context(cls, context: FormalContext, algorithm: tuple(CLOSURE_ITERATOR_REGISTRY) = 'CbO-FW', use_tqdm: bool = False):
        algo_func = CLOSURE_ITERATOR_REGISTRY[algorithm]
        attr_closure = lambda description: context.intent(context.extent(description))
        concepts = {Concept(frozenset(context.extent(intent)), frozenset(intent))
                    for intent in tqdm(algo_func(context.attributes, attr_closure), desc='Mine concepts', disable=not use_tqdm)}
        order = {(c1, c2) for c1 in concepts for c2 in concepts if c1[0] <= c2[0] }
        return cls(concepts, order)

    @property
    def intents(self) -> Poset:
        return self.__class__({c.intent for c in self}, {(c_big.intent, c_small.intent) for (c_small, c_big) in self.leq_order} )

    @property
    def extents(self) -> Poset:
        return self.__class__({c.extent for c in self}, {(c_small.extent, c_big.extent) for (c_small, c_big) in self.leq_order} )

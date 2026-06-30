import math
from typing import Protocol

from caspailleur.classes.poset import Poset, TElement



class PosetMeasureProtocol(Protocol):
    def __call__(
            self, poset: Poset,
            **kwargs
    ):
        ...

POSET_MEASURE_REGISTRY: dict[str, PosetMeasureProtocol] = dict()

def register_poset_measure(key: str):
    def decorator(func):
        #assert key not in LINE_LAYOUT_REGISTRY
        POSET_MEASURE_REGISTRY[key] = func
        return func

    return decorator


@register_poset_measure('is_boolean_lattice')
def is_boolean_lattice(poset: Poset) -> bool:
    n_atoms = math.log2(len(poset))
    if n_atoms != int(n_atoms):
        return False
    isomorphism = dict()
    bottom = poset.min()
    if bottom is None:
        return False
    isomorphism[bottom] = set()
    atoms = poset.direct_successors(bottom)
    if len(atoms) != n_atoms:
        return False
    for i, atom in enumerate(atoms):
        isomorphism[atom] = {i}

    elements = sorted(poset, key=lambda el: len(poset.predecessors(el)))
    for i, element in enumerate(elements):
        if element not in isomorphism: return False
        for other in elements[:i]:
            join = poset.supremum(element, other)
            if join not in isomorphism:
                isomorphism[join] = isomorphism[element] | isomorphism[other]
            if isomorphism[join] != isomorphism[element] | isomorphism[other]:
                return False

    return len(isomorphism) == len(poset)


@register_poset_measure('is_distributive_triplet')
def is_distributive_triplet(a: TElement, b: TElement, c: TElement, poset: Poset) -> bool:
    lhs = poset.infimum(a, poset.supremum(b, c))
    rhs = poset.supremum(poset.infimum(a, b), poset.infimum(a, c))
    return lhs == rhs


@register_poset_measure('is_distributive')
def is_distributive(poset: Poset) -> bool:
    elements = list(poset)
    for a in elements:
        for j, b in enumerate(elements[:-1]):
            for c in elements[j+1:]:
                if not is_distributive_triplet(a, b, c, poset):
                    return False
    return True


@register_poset_measure('is_meet_distributive_element')
def is_meet_distributive_element(a: TElement, poset: Poset) -> bool:
    meet = poset.infimum(a, *poset.direct_predecessors(a))

    interval = poset.successors(meet) & poset.predecessors(a)
    sublattice = Poset.from_functional_order(interval, leq_func=lambda x, y: (x, y) in poset)
    return is_boolean_lattice(sublattice)


@register_poset_measure('is_meet_distributive')
def is_meet_distributive(poset: Poset) -> bool:
    elements = list(poset)
    for element in elements:
        if not is_meet_distributive_element(element, poset):
            return False
    return True


@register_poset_measure('is_meet_semidistributive_triplet')
def is_meet_semidistributive_triplet(a: TElement, b: TElement, c: TElement, poset: Poset) -> bool:
    meet1 = poset.infimum(a, b)
    meet2 = poset.infimum(a, c)
    if meet1 != meet2:
        return True

    if poset.infimum(a, poset.supremum(b, c)) == meet1:
        return True
    return False


@register_poset_measure('meet_distributivity')
def meet_distributivity(poset: Poset) -> float:
    return sum(is_meet_distributive_element(el, poset) for el in poset)/len(poset)


@register_poset_measure('is_meet_semidistributive')
def is_meet_semidistributive(poset: Poset) -> bool:
    elements = list(poset)
    for a in elements:
        for j, b in enumerate(elements):
            for c in elements[j+1:]:
                if not is_meet_semidistributive_triplet(a, b, c, poset):
                    return False
    return True


@register_poset_measure('is_join_semidistributive_triplet')
def is_join_semidistributive_triplet(a: TElement, b: TElement, c: TElement, poset: Poset) -> bool:
    join1 = poset.supremum(a, b)
    join2 = poset.supremum(a, c)
    if join1 != join2:
        return True

    if poset.supremum(a, poset.infimum(b, c)) == join1:
        return True
    return False


@register_poset_measure('is_join_semidistributive')
def is_join_semidistributive(poset: Poset) -> bool:
    elements = list(poset)
    for a in elements:
        for j, b in enumerate(elements):
            for c in elements[j+1:]:
                if not is_join_semidistributive_triplet(a, b, c, poset):
                    return False
    return True

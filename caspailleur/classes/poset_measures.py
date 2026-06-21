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
    meets, joins = set(), set()
    elements = list(poset)
    for i, a in enumerate(elements):
        for b in elements[i + 1:]:
            meet = poset.infimum(a, b)
            if meet is None:  # i.e. meet does not exist
                return False
            if meet in {a, b}:  # implies "join in {a, b}"
                continue

            if meet in meets:
                return False
            meets.add(meet)

            join = poset.supremum(a, b)
            if join is None:  # i.e. join does not exist
                return False
            if join in joins:
                return False
            joins.add(join)
    return True

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
    sublattice = Poset.from_functional_order(interval, leq_func=lambda x, y: (x, y) in poset.leq_order)
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

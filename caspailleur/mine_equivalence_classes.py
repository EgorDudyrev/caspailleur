from typing import List, FrozenSet, Container, Dict, Collection, Iterator, Iterable

from .order import topological_sorting
from .base_functions import iset2ba, ba2iset

import numpy as np
import numpy.typing as npt
from skmine.itemsets import LCM
from bitarray import frozenbitarray as fbarray
from bitarray.util import zeros as bazeros


def list_intents_via_LCM(itemsets: List[Container[int]], min_supp: int = 1, n_attrs: int = None)\
        -> List[FrozenSet[int]]:
    lcm = LCM(min_supp=min_supp)
    itsets = lcm.fit_discover(itemsets)['itemset']

    n_attrs = max(max(itset) for itset in itemsets if itset) + 1 if n_attrs is None else n_attrs
    itsets = [iset2ba(itset, n_attrs) for itset in itsets]
    itsets = topological_sorting(itsets)[0]
    itsets = [frozenset(ba2iset(ba)) for ba in itsets]

    biggest_itset = frozenset(range(n_attrs))
    if itsets[0] != biggest_itset:
        itsets.insert(0, biggest_itset)

    smallest_itset = set(range(n_attrs))
    for itset in itsets:
        smallest_itset &= set(itset)
    if itsets[-1] != smallest_itset:
        itsets.append(frozenset(smallest_itset))
    return itsets


def list_attribute_concepts(intents: List[fbarray], parents: List[Collection[int]]) -> List[int]:
    """Get the indices of `intents` selected by each sole attribute"""
    attr_concepts = [-1] * len(intents[0])
    for intent_i, intent in enumerate(intents):
        new_attrs = intent
        for parent_i in parents[intent_i]:
            new_attrs = new_attrs & (~intents[parent_i])  # i.e. new_attrs \ parent_attrs
            if not new_attrs.any():
                break

        for attr_i in new_attrs.itersearch(1):
            attr_concepts[attr_i] = intent_i
    return attr_concepts


def iter_attribute_extents(K: npt.NDArray[np.bool_]) -> Iterator[fbarray]:
    return (fbarray(ext.tolist()) for ext in K.T)


def iter_equivalence_class(attribute_extents: List[fbarray], intent: List[int] = None) -> Iterator[FrozenSet[int]]:
    intent = range(len(attribute_extents)) if intent is None else intent
    intent = sorted(intent)
    intent_set = frozenset(intent)

    N_OBJS, N_ATTRS = len(attribute_extents[0]), len(attribute_extents)

    def conjunct_extent(bitarrays: Iterator[fbarray]) -> fbarray:
        res = ~bazeros(N_OBJS)
        for ext in bitarrays:
            res &= ext
            if not res.any():
                break

        return fbarray(res)

    total_extent = conjunct_extent((attribute_extents[m] for m in intent))
    stack = [[m] for m in intent[::-1]]

    yield intent_set
    while stack:
        attrs_to_remove = stack.pop(0)
        last_attr = attrs_to_remove[-1]
        attrs_to_eval = sorted(intent_set - set(attrs_to_remove))

        conj = conjunct_extent((attribute_extents[m] for m in attrs_to_eval))
        if conj != total_extent:
            continue

        # conj == total_extent
        yield frozenset(attrs_to_eval)
        stack += [attrs_to_remove+[m] for m in intent[::-1] if m > last_attr]


def list_keys_via_eqclass(equiv_class: Iterable[FrozenSet[int]]) -> List[FrozenSet[int]]:
    potent_keys = []
    for new_key in equiv_class:
        potent_keys = [key for key in potent_keys if new_key & key != new_key]
        potent_keys.append(new_key)
    return potent_keys


def list_passkeys_via_keys(keys: Iterable[FrozenSet[int]]) -> List[FrozenSet[int]]:
    passkeys = []
    for key in keys:
        if not passkeys or len(key) == len(passkeys[-1]):
            passkeys.append(key)
            continue

        if len(key) < len(passkeys[-1]):
            passkeys = [key]
    return passkeys

from typing import List, Dict, Iterator, Iterable

from .order import topological_sorting
from .base_functions import isets2bas, bas2isets

from skmine.itemsets import LCM
from bitarray import bitarray, frozenbitarray as fbarray
from bitarray.util import zeros as bazeros
from collections import deque


def list_intents_via_LCM(itemsets: List[fbarray], min_supp: float = 1) -> List[fbarray]:
    n_attrs = len(itemsets[0])

    lcm = LCM(min_supp=min_supp)
    intents = lcm.fit_discover(bas2isets(itemsets))['itemset']
    intents = list(isets2bas(intents, n_attrs))
    intents = topological_sorting(intents)[0]

    biggest_intent = ~bazeros(n_attrs)
    if intents[-1] != biggest_intent:
        intents.append(fbarray(biggest_intent))

    smallest_intent = ~bazeros(n_attrs)
    for itset in itemsets:
        smallest_intent &= itset
        if not smallest_intent.any():
            break
    if intents[0] != smallest_intent:
        intents.insert(0, fbarray(smallest_intent))
    return intents


def list_attribute_concepts(intents: List[fbarray]) -> List[int]:
    """Get the indices of `intents` selected by each sole attribute"""
    assert (len(a) <= len(b) for a, b in zip(intents, intents[1:])),\
        'The list of `intents` should be topologically sorted. With the cardinality minimal intent being at the start'

    attr_concepts = [-1] * len(intents[0])
    found_attrs = bazeros(len(intents[0]))
    for intent_i, intent in enumerate(intents):
        for m in intent.itersearch(True):
            if attr_concepts[m] == -1:
                attr_concepts[m] = intent_i
        found_attrs |= intent
        if found_attrs.all():
            break

    return attr_concepts


def iter_equivalence_class(attribute_extents: List[fbarray], intent: fbarray = None) -> Iterator[fbarray]:
    N_OBJS, N_ATTRS = len(attribute_extents[0]), len(attribute_extents)

    intent = bazeros(N_ATTRS) if intent is None else intent

    def conjunct_extent(premise: fbarray) -> fbarray:
        res = ~bazeros(N_OBJS)
        for m in premise.itersearch(True):
            res &= attribute_extents[m]
            if not res.any():
                break

        return fbarray(res)

    total_extent = conjunct_extent(intent)
    stack = [[m] for m in intent.itersearch(True)][::-1]

    yield intent
    while stack:
        attrs_to_remove = stack.pop(0)
        last_attr = attrs_to_remove[-1]

        attrs_to_eval = bitarray(intent)
        for m in attrs_to_remove:
            attrs_to_eval[m] = False
        attrs_to_eval = fbarray(attrs_to_eval)

        conj = conjunct_extent(attrs_to_eval)
        if conj != total_extent:
            continue

        # conj == total_extent
        yield attrs_to_eval
        stack += [attrs_to_remove+[m] for m in intent.itersearch(True) if m > last_attr][::-1]


def list_keys_via_eqclass(equiv_class: Iterable[fbarray]) -> List[fbarray]:
    potent_keys = []
    for new_key in equiv_class:
        potent_keys = [key for key in potent_keys if new_key & key != new_key]
        potent_keys.append(new_key)
    return potent_keys


def list_passkeys_via_eqclass(equiv_class: Iterable[fbarray]) -> List[fbarray]:
    passkeys = []
    for descr in equiv_class:
        if not passkeys or descr.count() == passkeys[-1].count():
            passkeys.append(descr)

        if descr.count() < passkeys[-1].count():
            passkeys = [descr]
    return passkeys


def list_keys(intents: List[fbarray], only_passkeys: bool = False) -> Dict[fbarray, int]:
    assert all(a.count() <= b.count() for a, b in zip(intents, intents[1:])), \
        'The `intents` list should be topologically sorted by ascending order'

    n_attrs, n_intents = len(intents[-1]), len(intents)
    attrs_descendants = [bazeros(n_intents) for _ in range(n_attrs)]
    for intent_i, intent in enumerate(intents):
        for m in intent.itersearch(True):
            attrs_descendants[m][intent_i] = True

    # assuming that every subset of a key is a key => extending not-a-key cannot result in a key
    # and every subset of a passkey is a passkey

    single_attrs = list(isets2bas([[m] for m in range(n_attrs)], n_attrs))

    if only_passkeys:
        passkey_sizes = [n_attrs] * n_intents

    keys_dict = {fbarray(bazeros(n_attrs)): 0}
    attrs_to_test = deque([m_ba for m_ba in single_attrs])
    while attrs_to_test:
        attrs = attrs_to_test.popleft()
        attrs_indices = list(attrs.itersearch(True))

        if any(attrs & (~single_attrs[m]) not in keys_dict for m in attrs_indices):
            continue

        max_attr_idx = attrs_indices[-1] if attrs_indices else -1
        common_descendants = ~bazeros(n_intents)
        for m in attrs.itersearch(True):
            common_descendants &= attrs_descendants[m]

        meet_intent_idx = common_descendants.find(True)

        if only_passkeys:
            # `attrs` is not a passkey because of the size
            if passkey_sizes[meet_intent_idx] < attrs.count():
                continue

        # if subset of attrs is not a key, or a key of the same intent
        if any(keys_dict[attrs & (~single_attrs[m])] == meet_intent_idx for m in attrs_indices):
            continue

        keys_dict[attrs] = meet_intent_idx
        if only_passkeys:
            passkey_sizes[meet_intent_idx] = attrs.count()
        if meet_intent_idx != n_intents-1:
            attrs_to_test.extend([attrs | m_ba for m_ba in single_attrs[max_attr_idx+1:]])

    return keys_dict


def list_passkeys(intents: List[fbarray]) -> Dict[fbarray, int]:
    return list_keys(intents, only_passkeys=True)

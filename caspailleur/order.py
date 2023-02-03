from typing import List, FrozenSet
from bitarray import bitarray, frozenbitarray as fbarray
from bitarray.util import zeros as bazeros


def topological_sorting(elements: List[FrozenSet[int]]) -> (List[List[FrozenSet[int]]], List[int]):
    ars_topsort = sorted(elements, key=lambda el: (len(el), sorted(el)))

    el_idx_map = {el: i for i, el in enumerate(ars_topsort)}
    orig_to_topsort_indices_map = [el_idx_map[el] for el in elements]
    return ars_topsort, orig_to_topsort_indices_map


def inverse_order(order: List[FrozenSet[int]]) -> List[FrozenSet[int]]:
    inversed = [list() for _ in order]
    for child, parents in enumerate(order):
        for parent in parents:
            inversed[parent].append(child)
    inversed = [frozenset(vs) for vs in inversed]
    return inversed


def inverse_order_ba(order: List[fbarray]) -> List[fbarray]:
    inversed = [bazeros(len(order[0])) for _ in order]
    for child, parents_ba in enumerate(order):
        for parent in parents_ba.itersearch(True):
            inversed[parent][child] = True
    inversed = [fbarray(children) for children in inversed]
    return inversed


def sort_intents_inclusion(intents: List[FrozenSet[int]]) -> List[FrozenSet[int]]:
    assert all(len(a) <= len(b) for a, b in zip(intents, intents[1:])), \
        'The `intents` list should be topologically sorted by ascending order'

    lattice = [frozenset()] * len(intents)
    trans_lattice = [frozenset()] * len(intents)
    all_attrs = frozenset(intents[-1])
    n_intents, n_attrs = len(intents), len(all_attrs)
    ba_ones = ~bazeros(n_intents)

    attrs_descendants = [~ba_ones for _ in range(n_attrs)]
    for intent_i, intent in enumerate(intents):
        for m in intent:
            attrs_descendants[m][intent_i] = True

    for intent_i, intent in enumerate(intents[::-1]):
        intent_i = n_intents - intent_i - 1

        common_descendants = ba_ones.copy()
        for m in intent:
            common_descendants &= attrs_descendants[m]

        children = {(common_descendants & attrs_descendants[m]).find(True) for m in all_attrs - intent}

        trans_children = set()
        for child in children:
            trans_children |= trans_lattice[child]
        trans_lattice[intent_i] = frozenset(children | trans_children)
        lattice[intent_i] = frozenset(children - trans_children)

    return lattice


def trans_close_relation(parents_list: List[FrozenSet[int]]) -> List[FrozenSet[int]]:
    assert all([max(parents_) < i for i, parents_ in enumerate(parents_list) if parents_]), \
        "`parents_list` relation should be defined on a list, topologically sorted by descending order." \
        " So all parents_list of element `i` should have smaller indices"

    trans_parents = []
    for i, parents_ in enumerate(parents_list):
        trans_pars = set(parents_)
        for parent in parents_:
            trans_pars |= trans_parents[parent]
        trans_parents.append(frozenset(trans_pars))
    return trans_parents


def trans_close_relation_ba(parents_list: List[fbarray]) -> List[fbarray]:
    assert all([not parents_[i:].any() for i, parents_ in enumerate(parents_list)]), \
        "`parents_list` relation should be defined on a list, topologically sorted by descending order." \
        " So all parents_list of element `i` should have smaller indices"

    trans_parents = []
    for i, parents_ in enumerate(parents_list):
        trans_pars = bitarray(parents_)
        for parent in parents_.itersearch(True):
            trans_pars |= trans_parents[parent]
        trans_parents.append(fbarray(trans_pars))
    return trans_parents


def drop_transitive_parents(parents_list: List[FrozenSet[int]]) -> List[FrozenSet[int]]:
    ancestors_list = [set(parents) for parents in parents_list]
    for el_i, parents in enumerate(parents_list[1:], start=1):
        for parent_i in parents:
            ancestors_list[el_i] |= ancestors_list[parent_i]

    new_parents_list = [set(parents) for parents in parents_list]
    for el_i, parents in enumerate(parents_list[1:], start=1):
        for parent_i in sorted(parents, reverse=True):
            if parent_i in new_parents_list[el_i]:
                new_parents_list[el_i] -= ancestors_list[parent_i]

    new_parents_list = [frozenset(parents) for parents in new_parents_list]
    return new_parents_list

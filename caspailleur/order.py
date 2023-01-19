from typing import Dict, List, Tuple, FrozenSet
from bitarray import frozenbitarray as fbarray
from bitarray.util import zeros as bazeros

from tqdm import tqdm


def topological_sorting(elements: List[fbarray]) -> Tuple[List[fbarray], List[int]]:
    """Sort the elements so that the first element is the biggest, the last is the smallest.

    The second output is the mapping from the original indices to the sorted ones
    """
    ars_topsort = sorted(elements, key=lambda el: (-el.count(), list(el.itersearch(1))))

    bitars_idx_map = {el: i for i, el in enumerate(ars_topsort)}
    orig_to_topsort_indices_map = [bitars_idx_map[el] for el in elements]
    return ars_topsort, orig_to_topsort_indices_map


def construct_spanning_tree(elements: List[fbarray]) -> List[int]:
    """Find a parent for each element. Elements should be topologically presorted"""
    parents = [None] * len(elements)
    for i, el in enumerate(elements[1:], start=1):
        for j in range(i-1, 0-1, -1):
            if el & elements[j] == el:
                parents[i] = j
                break
    return parents


def split_sptree_to_chains(span_tree_parents: List[int]) -> List[List[int]]:
    span_tree_parents_set = set(span_tree_parents)
    leafs = [i for i in range(len(span_tree_parents)) if i not in span_tree_parents_set]
    del span_tree_parents_set

    chains = []
    for leaf in leafs:
        node, chain = leaf, []
        while node is not None:
            chain.append(node)
            node = span_tree_parents[node]
        chains.append(chain[::-1])
    return chains


def find_parents_in_chains(chains: List[List[int]], elements: List[fbarray], use_tqdm: bool = False)\
        -> List[FrozenSet[int]]:
    parents_list = []
    for el in tqdm(elements, disable=not use_tqdm, desc='Iterating elements'):
        if el.all():
            parents_list.append(frozenset([]))
            continue

        parents = []
        for chain in chains:
            for other_i in chain:
                if elements[other_i] == el:
                    break
                if elements[other_i] & el != el:
                    break

                parent = other_i
            parents.append(parent)
        parents_list.append(frozenset(parents))
    return parents_list


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


def sort_ba_inclusion(bitarrays: List[fbarray], use_tqdm: bool = False) -> List[FrozenSet[int]]:
    """Take the list of bitarrays and output the parents_list dict (succeeding elements by consumption order)"""
    bitars_topsort, orig_topsort_imap = topological_sorting(bitarrays)
    span_tree_parents = construct_spanning_tree(bitars_topsort)
    chains_list = split_sptree_to_chains(span_tree_parents)
    parents_list_trans = find_parents_in_chains(chains_list, bitars_topsort, use_tqdm=use_tqdm)
    parents_list_final = drop_transitive_parents(parents_list_trans)

    topsort_orig_imap = {v: i for i, v in enumerate(orig_topsort_imap)}
    parents_orig = [frozenset(topsort_orig_imap[parent] for parent in parents_list_final[topsort_i])
                    for topsort_i in orig_topsort_imap]
    return parents_orig


def inverse_order(order: List[FrozenSet[int]]) -> List[FrozenSet[int]]:
    inversed = [list() for _ in order]
    for child, parents in enumerate(order):
        for parent in parents:
            inversed[parent].append(child)
    inversed = [frozenset(vs) for vs in inversed]
    return inversed


def sort_ba_subsumption(bitarrays: List[fbarray], use_tqdm: bool = False) -> List[FrozenSet[int]]:
    return inverse_order(sort_ba_inclusion(bitarrays, use_tqdm))


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

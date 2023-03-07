from typing import List
from bitarray import bitarray, frozenbitarray as fbarray
from bitarray.util import zeros as bazeros
from tqdm import tqdm


def topological_sorting(elements: List[fbarray]) -> (List[fbarray], List[int]):
    """Sort the list of `elements` by cardinality and lexicographical order

    In the output, the first element is the smallest one, the last one is the biggest

    Parameters
    ----------
    elements:
        List of bitarrays to sort (e.g. the list of intents)

    Returns
    -------
    elements_sorted: List[fbarray]
    orig_to_topsort_indices_map: List[int]
        THe mapping from the original indices to the indices of topologically sorted array.
    """
    ars_topsort = sorted(elements, key=lambda el: (el.count(), tuple(el.itersearch(True))))

    el_idx_map = {el: i for i, el in enumerate(ars_topsort)}
    orig_to_topsort_indices_map = [el_idx_map[el] for el in elements]
    return ars_topsort, orig_to_topsort_indices_map


def test_topologically_sorted(elements: List[fbarray]) -> bool:
    """Test if the list of `elements` is topologically sorted (from the smallest to the biggest element)

    One can obtain topologically sorted list of elements with ``topological_sorting(elements)`` function
    """
    return all(a.count() <= b.count() for a, b in zip(elements, elements[1:]))


def inverse_order(order: List[fbarray]) -> List[fbarray]:
    """Reverse the given order. So that new_order[i][j] = order[j][i]"""
    inversed = [bazeros(len(order[0])) for _ in order]
    for el_i, ordered in enumerate(order):
        for el_j in ordered.itersearch(True):
            inversed[el_j][el_i] = True

    return [fbarray(ordered) for ordered in inversed]


def sort_intents_inclusion(intents: List[fbarray], use_tqdm=False, return_transitive_order: bool = False)\
        -> List[fbarray] or (List[fbarray], List[fbarray]):
    """Returns indices of the smallest intents bigger than each intent. Optionally, return the transitive closure

    Returned is the list `lattice` so that intent with index `i` is included in intents with indices `lattice[i]`
    """
    assert test_topologically_sorted(intents), 'The `intents` list should be topologically sorted by ascending order'

    n_intents, n_attrs = len(intents), len(intents[0])
    zero_intents = fbarray(bazeros(n_intents))
    all_attrs = ~fbarray(bazeros(n_attrs))

    lattice = [fbarray(bazeros(n_intents))] * len(intents)
    trans_lattice = [fbarray(bazeros(n_intents))] * len(intents)

    attrs_descendants = [bitarray(zero_intents) for _ in range(n_attrs)]
    for intent_i, intent in enumerate(intents):
        for m in intent.itersearch(True):
            attrs_descendants[m][intent_i] = True

    for intent_i in tqdm(range(n_intents-1, -1, -1), disable=not use_tqdm, desc='Sorting intents'):
        intent = intents[intent_i]

        common_descendants = bitarray(~zero_intents)
        for m in intent.itersearch(True):
            common_descendants &= attrs_descendants[m]

        children = bitarray(zero_intents)
        for new_m in (all_attrs & ~intent).itersearch(True):
            meet_idx = (common_descendants & attrs_descendants[new_m]).find(True)
            children[meet_idx] = True

        trans_children = bitarray(zero_intents)
        for child in children.itersearch(True):
            trans_children |= trans_lattice[child]
        trans_lattice[intent_i] = fbarray(children | trans_children)
        lattice[intent_i] = fbarray(children & ~trans_children)

    if return_transitive_order:
        return lattice, trans_lattice
    return lattice


def close_transitive_subsumption(subsumption_list: List[fbarray]) -> List[fbarray]:
    """Transitive closure over subsumption operation

    So that in the returned list `trans_subsumption_list`,
    `trans_subsumption_list[i]` contains indices of all intents, smaller that intent with index `i`
    """
    assert all([max(subsumed.itersearch(True)) < i for i, subsumed in enumerate(subsumption_list) if subsumed.any()]), \
        "`subsumption_list` relation should be defined on a list, topologically sorted by descending order." \
        " So all subsumption_list of element `i` should have smaller indices"

    trans_subsumption_list = []
    for subsumed in subsumption_list:
        trans_subsumed = bitarray(subsumed)
        for el_i in subsumed.itersearch(True):
            trans_subsumed |= trans_subsumption_list[el_i]
        trans_subsumption_list.append(fbarray(trans_subsumed))
    return trans_subsumption_list


def open_transitive_subsumption(trans_subsumption_list: List[fbarray]) -> List[fbarray]:
    """Convert transitive closure of subsumption relation into the subsumption relation

    So that in the returned list `subsumption_list`,
    `subsumption_list[i]` contains indices of the biggest intents, smaller that intent with index `i`
    """
    assert all([
        max(subsumed.itersearch(True)) < i for i, subsumed in enumerate(trans_subsumption_list) if subsumed.any()
    ]),\
        "`trans_subsumption_list` relation should be defined on a list, topologically sorted by descending order." \
        " So all trans_subsumption_list of element `i` should have smaller indices"

    subsumption_list = []
    for trans_subsumed in trans_subsumption_list:
        subsumed = bitarray(trans_subsumed)
        for el_i in list(subsumed.itersearch(True))[::-1]:
            if subsumed[el_i]:
                subsumed &= ~trans_subsumption_list[el_i]
        subsumption_list.append(fbarray(subsumed))

    return subsumption_list


def drop_transitive_subsumption(subsumption_list: List[fbarray]) -> List[fbarray]:
    """Delete pairs with transitive relation from the given order

    If there are i,j,k such that all (subsumption_list[i][j], subsumption_list[j][k], subsumption_list[i,k]) are True,
    then set subsumption_list[i][k] = False.
    """
    return open_transitive_subsumption(close_transitive_subsumption(subsumption_list))

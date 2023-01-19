from typing import FrozenSet, List
from bitarray import frozenbitarray as fbarray


def linearity_index(parents: List[FrozenSet[int]], include_top_bottom: bool = True) -> float:
    assert all([max(parents_) < i for i, parents_ in enumerate(parents) if parents_]),\
        "`parents` relation should be defined on a list, topologically sorted by descending order." \
        " So all parents of element `i` should have smaller indices"

    n_elems = len(parents)
    trans_parents = [set(parents_) for parents_ in parents]
    for i, parents_ in enumerate(parents):
        for parent in parents_:
            trans_parents[i] |= trans_parents[parent]
    assert len(trans_parents[0]) == 0 and len(trans_parents[-1]) == n_elems - 1, \
        "Given `parents` relation should represent the lattice (i.e. have only one top and one bottom)"

    n_comparable = sum(len(tparents) for tparents in trans_parents)
    n_pairs = n_elems * (n_elems - 1) // 2

    if not include_top_bottom:
        # for (n_elems - 2) elements between top and bottom, drop pairs with the top,
        # for bottom element drop all (n_elems - 1) relations
        n_comparable -= (n_elems - 2) + (n_elems - 1)
        n_pairs -= 2*(n_elems-1) - 1

    return n_comparable / n_pairs


def distributivity_index(intents: List[fbarray], include_top_bottom: bool = True) -> float:
    intents_set = set(intents)

    n_nodes = len(intents)
    if include_top_bottom:
        indices_to_iterate = ((i, j) for i in range(0, n_nodes)     for j in range(i+1, n_nodes))
    else:
        indices_to_iterate = ((i, j) for i in range(1, n_nodes - 1) for j in range(i + 1, n_nodes - 1))

    n_distr, n_pairs = 0, 0
    for i, j in indices_to_iterate:
        n_distr += int(intents[i] | intents[j] in intents_set)
        n_pairs += 1

    return n_distr / n_pairs

from typing import FrozenSet, List
from bitarray import frozenbitarray as fbarray


def linearity_index(parents: List[FrozenSet[int]], include_top_bottom: bool = True) -> float:
    parents_trans = {k: set(vs) for k, vs in enumerate(parents)}
    for child in sorted(parents_trans):
        for parent in list(parents_trans[child]):
            parents_trans[child] |= parents_trans[parent]

    n_nodes = len(parents_trans)
    if include_top_bottom:
        indices_to_iterate = ((i, j) for i in range(0, n_nodes) for j in range(i+1, n_nodes))
    else:
        indices_to_iterate = ((i, j) for i in range(1, n_nodes - 1) for j in range(i+1, n_nodes - 1))

    n_comparable = 0
    n_pairs = 0
    for i, j in indices_to_iterate:
        n_comparable += int(i in parents_trans[j])
        n_pairs += 1

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

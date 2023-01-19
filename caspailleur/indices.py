from typing import FrozenSet, List


def linearity_index(transitive_parents: List[FrozenSet[int]], include_top_bottom: bool = True) -> float:
    assert all([max(parents_) < i for i, parents_ in enumerate(transitive_parents) if parents_]),\
        "`transitive_parents` relation should be defined on a list, topologically sorted by descending order." \
        " So all parents of element `i` should have smaller indices"

    n_elems = len(transitive_parents)
    n_comparable = sum(len(tparents) for tparents in transitive_parents)
    n_pairs = n_elems * (n_elems - 1) // 2

    if not include_top_bottom:
        assert len(transitive_parents[0]) == 0 and len(transitive_parents[-1]) == len(transitive_parents) - 1, \
            "Given `transitive_parents` relation should represent the lattice (i.e. have only one top and one bottom)"

        # for (n_elems - 2) elements between top and bottom, drop pairs with the top,
        # for bottom element drop all (n_elems - 1) relations
        n_comparable -= (n_elems - 2) + (n_elems - 1)
        n_pairs -= 2*(n_elems-1) - 1

    return n_comparable / n_pairs


def distributivity_index(
        intents: List[FrozenSet[int]], transitive_parents: List[FrozenSet[int]],
        include_top_bottom: bool = True
) -> float:
    assert all(len(a) <= len(b) for a, b in zip(intents, intents[1:])), \
        'The `intents` list should be topologically sorted by ascending order'
    assert (len(transitive_parents[0]) == 0) and len(transitive_parents[-1]) == len(transitive_parents)-1, \
        "Given `transitive_children` relation should represent the lattice (i.e. have only one top and one bottom)"

    n_intents = len(intents)
    intents_set = set(intents)

    n_distr = sum(len(tparents) for tparents in transitive_parents)
    n_distr += sum(
        intents[i] | intents[j] in intents_set
        for i in range(n_intents) for j in range(i+1, n_intents) if i not in transitive_parents[j]
    )

    n_pairs = n_intents * (n_intents - 1) // 2
    if not include_top_bottom:
        n_distr -= (n_intents - 1) + (n_intents - 2)  # dropping (n-1) rels for top intent and 1 rel. for (n-2) intents
        n_pairs -= 2 * (n_intents - 1) - 1

    return n_distr / n_pairs

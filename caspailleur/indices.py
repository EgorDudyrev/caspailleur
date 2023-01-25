from typing import FrozenSet, List
from collections import deque
from bitarray import frozenbitarray as fbarray
from tqdm import tqdm


def linearity_index(n_trans_parents: int, n_elements: int, include_top_bottom: bool = True) -> float:
    n_comparable = n_trans_parents
    n_pairs = n_elements * (n_elements - 1) // 2

    if not include_top_bottom:
        # for (n_elems - 2) elements between top and bottom, drop pairs with the top,
        # for bottom element drop all (n_elems - 1) relations
        n_comparable -= 2*n_elements - 3
        n_pairs -= 2*n_elements - 3

    if n_comparable > n_pairs:
        raise ValueError('Linearity index is computed in a wrong way (There should be problem with the code)')

    return n_comparable / n_pairs if n_pairs else 0


def distributivity_index(
        intents: List[fbarray], parents: List[FrozenSet[int]], n_trans_parents: int,
        include_top_bottom: bool = True, use_tqdm: bool = False
) -> float:
    assert all(len(a) <= len(b) for a, b in zip(intents, intents[1:])), \
        'The `intents` list should be topologically sorted by ascending order'

    n_distr = n_trans_parents

    for intent_idx, intent in tqdm(
            enumerate(intents), total=len(intents),
            disable=not use_tqdm, desc='enumerate intents'
    ):
        distr_ancestors = deque([
            (mother, father)
            for mother in parents[intent_idx] for father in parents[intent_idx]
            if mother < father and intents[mother] | intents[father] == intent
        ])

        visited_pairs = set()
        while distr_ancestors:
            pair = distr_ancestors.popleft()
            if pair in visited_pairs or pair[::-1] in visited_pairs:
                continue
            visited_pairs.add(pair)

            n_distr += 1

            mother, father = pair
            mother_intent, father_intent = intents[mother], intents[father]

            distr_ancestors.extend([
                (mother, gfather) for gfather in parents[father]
                if mother_intent | intents[gfather] == intent
            ])
            distr_ancestors.extend([
                (gmother, father) for gmother in parents[mother]
                if intents[gmother] | father_intent == intent
            ])

    n_intents = len(intents)
    n_pairs = n_intents * (n_intents - 1) // 2
    if not include_top_bottom:
        n_distr -= 2*n_intents - 3  # dropping (n-1) rels for top intent and 1 rel. for (n-2) intents
        n_pairs -= 2 * n_intents - 3

    if n_distr > n_pairs:
        raise ValueError('Distributivity index is computed in a wrong way (There should be problem with the code)')

    return n_distr / n_pairs if n_pairs else 0


def distributivity_index_ba(
        intents: List[fbarray], parents: List[fbarray], n_trans_parents: int,
        include_top_bottom: bool = True, use_tqdm: bool = False
) -> float:
    assert all(len(a) <= len(b) for a, b in zip(intents, intents[1:])), \
        'The `intents` list should be topologically sorted by ascending order'

    n_distr = n_trans_parents

    for intent_idx, intent in tqdm(
            enumerate(intents), total=len(intents),
            disable=not use_tqdm, desc='enumerate intents'
    ):
        distr_ancestors = deque([
            (mother, father)
            for mother in parents[intent_idx].itersearch(True) for father in parents[intent_idx].itersearch(True)
            if mother < father and intents[mother] | intents[father] == intent
        ])

        visited_pairs = set()
        while distr_ancestors:
            pair = distr_ancestors.popleft()
            if pair in visited_pairs or pair[::-1] in visited_pairs:
                continue
            visited_pairs.add(pair)

            n_distr += 1

            mother, father = pair
            mother_intent, father_intent = intents[mother], intents[father]

            distr_ancestors.extend([
                (mother, gfather) for gfather in parents[father].itersearch(True)
                if mother_intent | intents[gfather] == intent
            ])
            distr_ancestors.extend([
                (gmother, father) for gmother in parents[mother].itersearch(True)
                if intents[gmother] | father_intent == intent
            ])

    n_intents = len(intents)
    n_pairs = n_intents * (n_intents - 1) // 2
    if not include_top_bottom:
        n_distr -= 2 * n_intents - 3  # dropping (n-1) rels for top intent and 1 rel. for (n-2) intents
        n_pairs -= 2 * n_intents - 3

    if n_distr > n_pairs:
        raise ValueError('Distributivity index is computed in a wrong way (There should be problem with the code)')

    return n_distr / n_pairs if n_pairs else 0

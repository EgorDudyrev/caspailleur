from typing import List, Dict, Tuple, Iterator, Iterable
from bitarray import frozenbitarray as fbarray, bitarray
from tqdm import tqdm
from caspailleur.order import test_topologically_sorted


def saturate(premise: fbarray, impls: List[Tuple[fbarray, int]], intents: List[fbarray]) -> fbarray:
    """Extend `premise` with implications from `impl` and intents `intents`

    Parameters
    ----------
    premise: frozenbitarray
        bitarray to saturate with implications (`impls`) and intents
    impls:
        List of implications to saturate with. Each implication is a pair of a bitarray and an intent index.
        Intent serves as the conclusion of each implication.
    intents: List[frozenbitarray]
        List of intents

    Returns
    -------
    saturated_premise: frozenbitarray
        `Premise` saturated with implications and intents


    Examples
    --------
    intents = [ bitarray('0000'), bitarray('0110'), bitarray('1111') ]
    impls = [ (bitarray('0100'), 1),  (bitarray('0001'), 2) ]

    Then premise '0101' contains the left part of the first implication, so it should be extended by intent #1: '0111'.
    New premise '0111' contains the left part of the second implication, so it should be extended by intent #2: '1111'.

    saturate( bitarray('0101'), impls, intents ) --> bitarray('0101') | intents[1] | intents[2] = bitarray('1111')
    """
    new_closure, old_closure = bitarray(premise), None
    new_unused_impl, old_unused_impl = list(impls), None

    while old_closure != new_closure:
        old_closure, old_unused_impl = bitarray(new_closure), new_unused_impl
        new_unused_impl = []

        for old_prem, old_closure_i in old_unused_impl:
            if old_prem & new_closure == old_prem:
                new_closure |= intents[old_closure_i]
            else:
                new_unused_impl.append((old_prem, old_closure_i))
    return fbarray(new_closure)


def test_if_proper_premise_via_keys(
        key: fbarray, intent_idx: int, intents: List[fbarray], keys_prevsize: Dict[fbarray, int]
) -> bool:
    """Test if `key` is a proper premise given dict of keys of smaller size

    Parameters
    ----------
    key: frozenbitarray
        A key (bitarray) to test
    intent_idx: int
        Index of the intent which is closure of `key`
    intents: List[frozenbitarray]
        List of bitarrays representing intents (i.e. closed subset of attributes)
    keys_prevsize: Dict[frozenbitarray, int]
        Dictionary containing all keys of smaller sizes and intents, corresponding to them.
        Passing dictionary of keys of size = size(key) - 1 is enough for algorithm to work.

    Returns
    -------
    flg: bool
        A flag, whether `key` is a proper premise

    Size of key -- size(key) -- is the number of True values in it (i.e. key.count() if key is a bitarray).
    """
    intent = intents[intent_idx]
    if key == intent:
        return False

    if key.count() == 0:
        return True

    cumulative_key = bitarray(key)
    for m in key.itersearch(True):
        prekey = bitarray(key)
        prekey[m] = False

        cumulative_key |= intents[keys_prevsize[fbarray(prekey)]]
        if cumulative_key == intent:
            return False
    return True


def iter_proper_premises_via_keys(intents: List[fbarray], keys_to_intents: Dict[fbarray, int])\
        -> Iterator[Tuple[fbarray, int]]:
    """Obtain the set of proper premises given intents, intents parents relation, and keys

    Parameters
    ----------
    intents: List[frozenbitarray]
        list of closed descriptions (in the form of bitarrays)
    keys_to_intents: Dict[frozenbitarray, int]
        the dictionary of keys in the context and the indices of the corresponding intents

    Returns
    -------
    proper_premises: Iterator[Tuple[frozenbitarray, int]]
        Iterator with found proper premises

    """
    return (
        (key, intent_idx) for key, intent_idx in keys_to_intents.items()
        if test_if_proper_premise_via_keys(key, intent_idx, intents, keys_to_intents)
    )


def list_pseudo_intents_via_keys(
        keys_intent_map: Iterable[Tuple[fbarray, int]], intents: List[fbarray],
        use_tqdm: bool = False, n_keys: int = None
) -> List[Tuple[fbarray, int]]:
    """List pseudo-intents (and indices of their intents) based on keys (and indices of their intents)

    Parameters
    ----------
    keys_intent_map: Iterable[Tuple[frozenbitarray, int]]
        The list (or generator) of keys and intents indices, corresponding to them
    intents: List[frozenbitarray]
        The list of intents (i.e. closed subsets of attributes)
    use_tqdm: bool
        A flag, whether to visualize the algorithm progress with tqdm progressbar
    n_keys: int
        Total number of keys (for better tqdm progressbar visualisation)

    Returns
    -------
    pseudo_intents_map: List[Tuple[frozenbitarray, int]]
        List of pseudo-intents and the indices of corresponding intents

    """
    assert test_topologically_sorted(intents), 'The `intents` list should be topologically sorted by ascending order'

    def add_pintent(
            new_key: fbarray, new_pintent: fbarray, new_intent_i: int,
            pintents_list: List[Tuple[fbarray, fbarray, int]]
    ) -> List[Tuple[fbarray, fbarray, int]]:
        if not pintents_list or pintents_list[-1][1].count() < new_pintent.count():
            pintents_list.append((new_key, new_pintent, new_intent_i))
            return pintents_list

        for new_idx in range(len(pintents_list)-1, -1, -1):
            if pintents_list[new_idx][1] == new_pintent:
                return pintents_list
            if pintents_list[new_idx-1][1].count() < new_pintent.count():
                break
        pintents_list.insert(new_idx, (new_key, new_pintent, new_intent_i))

        for idx in range(new_idx, len(pintents_list)):
            if len(pintents_list) <= idx:
                break

            new_pintents = []
            for (pi_key, pintent, pi_intent_idx) in pintents_list[idx:]:
                resaturated = saturate(pi_key, [pi_data[1:] for pi_data in pintents_list[:idx]], intents)
                if resaturated != intents[pi_intent_idx]:
                    new_pintents.append((pi_key, resaturated, pi_intent_idx))
            pintents_list[idx:] = sorted(new_pintents, key=lambda pi_data: pi_data[1].count())

        return pintents_list

    pseudo_intents: List[Tuple[fbarray, fbarray, int]] = []
    for key, intent_i in tqdm(keys_intent_map, total=n_keys, disable=not use_tqdm, desc="Iterate p.intent candidates"):
        key_saturated = saturate(key, [pi_data[1:] for pi_data in pseudo_intents], intents)
        if key_saturated == intents[intent_i]:
            continue

        pseudo_intents = add_pintent(key, key_saturated, intent_i, pseudo_intents)

    return [tuple(pi_data[1:]) for pi_data in pseudo_intents]

from typing import List, Dict, Tuple, Iterator
from bitarray import frozenbitarray as fbarray, bitarray
from tqdm import tqdm
from caspailleur.order import test_topologically_sorted


def saturate(premise: fbarray, impls: List[Tuple[fbarray, int]], intents: List[fbarray]) -> fbarray:
    """Extend `premise` with implications from `impl`"""
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


def iter_proper_premises_via_keys(intents: List[fbarray], keys_to_intents: Dict[fbarray, int]) -> Iterator[fbarray]:
    """Obtain the set of proper premises given intents, intents parents relation, and keys

    Parameters
    ----------
    intents: list of closed descriptions (in the form of binary attributes)
    keys_to_intents: the dictionary of keys in the context and the indices of the corresponding intents
    """
    return (
        key for key, intent_idx in keys_to_intents.items()
        if test_if_proper_premise_via_keys(key, intent_idx, intents, keys_to_intents)
    )


def list_pseudo_intents_via_keys(
        keys_intent_map: Iterator[Tuple[fbarray, int]], intents: List[fbarray],
        use_tqdm: bool = False, n_keys: int = None
) -> List[Tuple[fbarray, int]]:
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

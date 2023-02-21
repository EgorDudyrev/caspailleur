from typing import List, FrozenSet, Dict, Tuple, Iterator
from bitarray import frozenbitarray as fbarray, bitarray
from bitarray.util import zeros as bazeros
from tqdm import tqdm

FSetInt = FrozenSet[int]


def iter_proper_premises_via_keys(intents: List[FSetInt], keys_to_intents: Dict[FSetInt, int]) -> Iterator[FSetInt]:
    """Obtain the set of proper premises given intents, intents parents relation, and keys

    Parameters
    ----------
    intents: list of closed descriptions (in the form of binary attributes)
    keys_to_intents: the dictionary of keys in the context and the indices of the corresponding intents
    """
    for key in keys_to_intents:
        intent = intents[keys_to_intents[key]]
        if key == intent:
            continue

        cumulative_key = set(key)
        for n in key:
            cumulative_key |= intents[keys_to_intents[key - {n}]]
            if cumulative_key == intent:
                break
        else:  # if after the cycle cum_intent still != intent
            yield key


def iter_proper_premises_via_keys_ba(intents: List[fbarray], keys_to_intents: Dict[fbarray, int]) -> Iterator[fbarray]:
    """Obtain the set of proper premises given intents, intents parents relation, and keys

    Parameters
    ----------
    intents: list of closed descriptions (in the form of binary attributes)
    keys_to_intents: the dictionary of keys in the context and the indices of the corresponding intents
    """
    for key in keys_to_intents:
        intent = intents[keys_to_intents[key]]
        if key == intent:
            continue

        cumulative_key = bitarray(key).copy()
        for m in key.itersearch(True):
            prekey = bitarray(key).copy()
            prekey[m] = False
            cumulative_key |= intents[keys_to_intents[fbarray(prekey)]]
            if cumulative_key == intent:
                break
        else:  # if after the cycle cum_intent still != intent
            yield key


def test_if_proper_premise_via_keys_ba(
        key: fbarray, intent_idx: int,
        intents: List[fbarray], keys_prevsize: Dict[fbarray, int]
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


def list_pseudo_intents_incremental(attribute_extents: List[FSetInt], intents: List[FSetInt], use_tqdm: bool = False) \
        -> List[FSetInt]:
    """From S. Obiedkov V. Duquenne paper of 2007"""
    ImplicationType = Tuple[FSetInt, FSetInt, FSetInt]  # (extent, premise, consequence)
    ConceptType = Tuple[FSetInt, FSetInt]  # (extent, intent)

    assert all(len(a) <= len(b) for a, b in zip(intents, intents[1:])), \
        'The `intents` list should be topologically sorted by ascending order'

    N_OBJS = max(max(extent) for extent in attribute_extents if extent) + 1
    N_ATTRS = len(attribute_extents)
    N_INTENTS = len(intents)
    attrs_descendants = [bazeros(N_INTENTS) for _ in range(N_ATTRS)]
    for intent_i, intent in enumerate(intents):
        for m in intent:
            attrs_descendants[m][intent_i] = True

    def saturate(new_prem: FSetInt, impl: Tuple[ImplicationType, ...]) -> FSetInt:
        """Extend `new_prem` with implications from `impl`"""
        new_closure, old_closure = new_prem, None
        new_unused_impl, old_unused_impl = list(impl), None

        while old_closure != new_closure:
            old_closure, old_unused_impl = set(new_closure), new_unused_impl
            new_unused_impl = []

            for (ext, prem, cons) in old_unused_impl:
                if prem < new_closure:  # if prem is a subset of new_closure
                    new_closure = new_closure | cons
                else:
                    new_unused_impl.append((ext, prem, cons))

        return frozenset(new_closure)

    def process_stable_concept(y: int, concept: ConceptType, new_stable_impls: Tuple[ImplicationType, ...]) \
            -> ConceptType or ImplicationType or None:
        new_ext = concept[0] & attribute_extents[y]
        new_prem = concept[1] | {y}

        common_intents = ~bazeros(N_INTENTS)
        for m_ in new_prem:
            common_intents &= attrs_descendants[m_]
        full_intent = intents[common_intents.find(True)]
        new_cons = {m_ for m_ in full_intent if m_ <= y}

        new_elem = new_stable_impl = None
        if new_cons == new_prem:
            new_elem = (new_ext, new_prem)
        else:
            if new_prem == saturate(new_prem, new_stable_impls):
                new_elem = new_stable_impl = (new_ext, new_prem, new_cons)

        return new_elem, new_stable_impl

    def process_modified_implication(y: int, implication: ImplicationType, min_mod_impls: Tuple[ImplicationType, ...])\
            -> (ImplicationType or None, ImplicationType or None, ImplicationType or None):
        old_prem = implication[1]

        is_min_impl = not any(min_prem <= old_prem for (_, min_prem, _) in min_mod_impls)
        new_prem = old_prem if is_min_impl else old_prem | {y}
        new_impl = (implication[0], new_prem, implication[2] | {y})

        if is_min_impl:
            min_mod_impl, non_min_mod_impl, el_to_filter = new_impl, None, implication
        else:
            min_mod_impl, non_min_mod_impl, el_to_filter = None, new_impl, None
        return min_mod_impl, non_min_mod_impl, el_to_filter

    def process_modified_concept(y: int, concept: ConceptType, min_mod_impl: Tuple[ImplicationType, ...])\
            -> (ConceptType, ImplicationType or None, ImplicationType or None):
        old_extent, old_intent = concept
        new_intent = old_intent | {y}

        is_min_impl = not any(min_prem <= old_intent for (_, min_prem, _) in min_mod_impl)
        new_min_impl = (old_extent, old_intent, new_intent) if is_min_impl else None

        new_concept = (old_extent, new_intent)
        return new_concept, new_min_impl, new_min_impl

    def fuse(basis: Tuple[ImplicationType, ...], extra_impl: Tuple[ImplicationType, ...]) -> List[ImplicationType]:
        n_extra_impl = len(extra_impl)
        extra_basis, extra_elements = [], []
        for impl_i, (ext, prem, cons) in enumerate(extra_impl):
            other_impl = basis + extra_basis
            if impl_i + 1 < n_extra_impl:
                other_impl += extra_impl[impl_i+1:]

            prem_satur = saturate(prem, other_impl)
            if (prem_satur == cons) or not (prem_satur <= cons):
                continue
            # if prem_satur is a proper subset of cons
            extra_basis.append((ext, prem_satur, cons))
            extra_elements.append((ext, prem_satur, cons))

        return extra_elements

    def add_attribute(attr_idx: int, elements: Tuple[ImplicationType or ConceptType, ...]) \
            -> List[ImplicationType or ConceptType]:
        """

        Input
        -----
        attr_idx: new attribute
        y_extent: extent of attr_idx
        elements:
        N: list of all attributes already processed

        Output
        ------
        `Elements` consists of all concepts of L(K_y) and implications of B(K_y)
        """
        old_stable_impls, new_stable_impls, min_mod_impls, non_min_mod_impls, mod_concepts = [], [], [], [], []
        filter_elements, append_elements = [], []

        for el_idx, element in enumerate(elements, start=1):
            to_modify = element[0] <= attribute_extents[attr_idx]  # element extent is a subset of attr_idx'
            is_concept = len(element) == 2  # element is a concept, i.e. it contains (extent, intent)

            el_to_filter, el_to_append = element, None
            old_stable_impl, new_stable_impl, min_mod_impl, non_min_mod_impl, mod_concept = None, None, None, None, None

            if to_modify and is_concept:
                mod_concept, min_mod_impl, el_to_filter = process_modified_concept(attr_idx, element, min_mod_impls)
            if to_modify and not is_concept:
                min_mod_impl, non_min_mod_impl, el_to_filter = process_modified_implication(attr_idx, element, min_mod_impls)
            if not to_modify and is_concept:
                el_to_append, new_stable_impl = process_stable_concept(attr_idx, element, new_stable_impls)
            if not to_modify and not is_concept:
                old_stable_impl = element

            # Putting new elements to lists
            for ar, el in [
                (filter_elements, el_to_filter), (append_elements, el_to_append),
                (old_stable_impls, old_stable_impl), (new_stable_impls, new_stable_impl),
                (min_mod_impls, min_mod_impl), (non_min_mod_impls, non_min_mod_impl), (mod_concepts, mod_concept)
            ]:
                if el:
                    ar.append(el)

        fused_elements = fuse(old_stable_impls + new_stable_impls + min_mod_impls, non_min_mod_impls)
        extra_elements = sorted(mod_concepts + fused_elements, key=lambda elem: elem[-1], reverse=False)
        new_elements = filter_elements + append_elements + extra_elements

        return new_elements

    elements_final: List[ConceptType or ImplicationType] = [(frozenset(range(N_OBJS)), frozenset([]))]
    for y in tqdm(range(N_ATTRS), desc='Incrementing attributes', disable=not use_tqdm):
        elements_final = add_attribute(y, elements_final)

    return [elem[1] for elem in elements_final if len(elem) == 3]  # return only the implications


def list_pseudo_intents_via_keys(
        keys_intent_map: Iterator[Tuple[FSetInt, int]], intents: Tuple[FSetInt, ...],
        use_tqdm: bool = False, n_keys=None
) -> List[Tuple[FSetInt, FSetInt, int]]:
    assert all(len(a) <= len(b) for a, b in zip(intents, intents[1:])), \
        'The `intents` list should be topologically sorted by ascending order'

    def saturate(new_prem: FSetInt, impls: List[Tuple[FSetInt, FSetInt, int]], intents_: List[FSetInt]) -> FSetInt:
        """Extend `new_prem` with implications from `impl`"""
        new_closure, old_closure = set(new_prem), None
        new_unused_impl, old_unused_impl = list(impls), None

        while old_closure != new_closure:
            old_closure, old_unused_impl = set(new_closure), new_unused_impl
            new_unused_impl = []

            for (old_key, old_prem, old_closure_i) in old_unused_impl:
                if not old_prem < new_closure:
                    new_unused_impl.append((old_key, old_prem, old_closure_i))
                    continue
                new_closure |= intents_[old_closure_i]

        return frozenset(new_closure)

    def add_pintent(
            new_key: FSetInt, new_pintent: FSetInt, new_intent_i: int,
            pintents_list: List[Tuple[FSetInt, FSetInt, int]]
    ) -> List[Tuple[FSetInt, FSetInt, int]]:

        if not pintents_list or len(pintents_list[-1][1]) < len(new_pintent):
            pintents_list.append((new_key, new_pintent, new_intent_i))
            return pintents_list

        for new_idx in range(len(pintents_list)-1, -1, -1):
            if pintents_list[new_idx][1] == new_pintent:
                return pintents_list
            if len(pintents_list[new_idx-1][1]) < len(new_pintent):
                break
        pintents_list.insert(new_idx, (new_key, new_pintent, new_intent_i))

        for idx in range(new_idx, len(pintents_list)):
            if len(pintents_list) <= idx:
                break

            new_pintents = ((pi_key, saturate(pi_key, pintents_list[:idx], intents), pi_intent_idx)
                            for pi_key, pintent, pi_intent_idx in pintents_list[idx:])
            new_pintents = ((pi_key, pintent, intent_idx) for pi_key, pintent, intent_idx in new_pintents
                            if pintent != intents[intent_idx])

            pintents_list[idx:] = sorted(new_pintents, key=lambda pi_data: len(pi_data[1]))

        return pintents_list

    pseudo_intents: List[Tuple[FSetInt, FSetInt, int]] = []
    for key, intent_i in tqdm(keys_intent_map, total=n_keys, disable=not use_tqdm, desc="Iterate possible P.Intents"):
        key_saturated = saturate(key, pseudo_intents, intents)
        if key_saturated == intents[intent_i]:
            continue

        pseudo_intents = add_pintent(key, key_saturated, intent_i, pseudo_intents)

    return pseudo_intents

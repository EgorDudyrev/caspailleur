from typing import List, FrozenSet, Dict, Tuple, Iterator

from bitarray import frozenbitarray as fbarray, bitarray as barray
from bitarray.util import zeros as bazeros
from tqdm import tqdm

from .base_functions import iset2ba, ba2iset


def iter_proper_premises_via_keys(
        intents: List[fbarray],
        keys_to_intents: Dict[fbarray, int]
) -> Iterator[fbarray]:
    """Obtain the set of proper premises given intents, intents parents relation, and keys

    Parameters
    ----------
    intents: list of closed descriptions (in the form of binary attributes)
    parents: parent relation among intents (defined by indexes of `intents` list)
    keys_to_intents: the dictionary of keys in the context and the indices of the corresponding intents
    """
    n_attrs = len(intents[0])
    single_attr_negations = [~iset2ba([i], n_attrs) for i in range(n_attrs)]

    for key in keys_to_intents:
        intent = intents[keys_to_intents[key]]
        if key == intent:
            continue

        cumulative_key = barray(key)
        for n in key.itersearch(1):
            prekey = key & single_attr_negations[n]

            cumulative_key |= intents[keys_to_intents[prekey]]
            if cumulative_key == intent:
                break
        else:  # if after the cycle cum_intent still != intent
            yield key


def list_pseudo_intents_incremental(attribute_extents: List[fbarray], use_tqdm: bool = False, verbose: bool = False) \
        -> List[Tuple[fbarray, fbarray, fbarray]]:
    """From S. Obiedkov V. Duquenne paper of 2007"""
    N_OBJS, N_ATTRS = len(attribute_extents[0]), len(attribute_extents)

    ImplicationType = Tuple[fbarray, fbarray, fbarray]  # (extent, premise, consequence)
    ConceptType = Tuple[fbarray, fbarray]  # (extent, intent)

    def ba_tuple_to_str(ba_tuple: ImplicationType or ConceptType) -> str:
        idx_names = 'abcdefghijklmnopqrstuvwxyz'
        idx_numbs = '123456789'
        idxs_verb = [''.join([idx_names[idx] if ba_idx == 0 else idx_numbs[idx] for idx in set(ba2iset(ba))])
                     if ba.count() else r"∅"
                     for ba_idx, ba in enumerate(ba_tuple)]
        return '(' + ', '.join(idxs_verb) + ')'

    def saturate(new_prem: fbarray, impl: Tuple[ImplicationType, ...]) -> fbarray:
        new_closure, old_closure = new_prem, None
        new_unused_impl, old_unused_impl = list(impl), None

        while old_closure != new_closure:
            old_closure, old_unused_impl = new_closure, new_unused_impl
            new_unused_impl = []

            for (ext, prem, cons) in old_unused_impl:
                if prem & new_closure == prem:  # if prem is a subset of new_closure
                    new_closure = new_closure | cons
                else:
                    new_unused_impl.append((ext, prem, cons))

        return fbarray(new_closure)

    def process_stable_concept(y: int, concept: ConceptType, new_stable_impl: Tuple[ImplicationType, ...])\
            -> (ConceptType or None, ImplicationType or None):
        print('Stable concept') if verbose else None
        extent, intent = concept
        new_ext = extent & attribute_extents[y]
        new_prem = intent | iset2ba([y], N_ATTRS)
        new_cons = iset2ba([n for n in range(y+1) if new_ext & attribute_extents[n] == new_ext], N_ATTRS)

        new_concept, new_impl = None, None
        if new_cons == new_prem:
            new_concept = (new_ext, new_prem)
        else:
            if new_prem == saturate(new_prem, new_stable_impl):
                new_impl = (new_ext, new_prem, new_cons)

        return new_concept, new_impl

    def process_modified_implication(y: int, implication: ImplicationType, min_mod_impl: Tuple[ImplicationType, ...]):
        print('Modified implication') if verbose else None
        ext, prem, cons = implication
        new_cons = cons | iset2ba([y], N_ATTRS)

        is_min_impl = all(min_prem & prem != min_mod_impl for (_, min_prem, _) in min_mod_impl)
        new_prem = prem if is_min_impl else prem | iset2ba([y], N_ATTRS)
        new_impl = (ext, new_prem, new_cons)
        return new_impl, is_min_impl

    def process_modified_concept(y: int, concept: ConceptType, min_mod_impl: Tuple[ImplicationType, ...])\
            -> (ConceptType, ImplicationType or None):
        print('Modified concept') if verbose else None
        extent, intent = concept
        new_concept = (extent, intent | iset2ba([y], N_ATTRS))

        new_min_impl = None
        if all(min_prem & intent != min_prem for (_, min_prem, _) in min_mod_impl):
            new_min_impl = (extent, intent, intent | iset2ba([y], N_ATTRS))

        return new_concept, new_min_impl

    def fuse(basis: Tuple[ImplicationType, ...], extra_impl: Tuple[ImplicationType, ...]) -> List[ImplicationType]:
        print('Fuse') if verbose else None

        n_extra_impl = len(extra_impl)
        extra_basis, extra_elements = [], []
        for impl_i, (ext, prem, cons) in enumerate(extra_impl):
            other_impl = basis + extra_basis
            if impl_i + 1 < n_extra_impl:
                other_impl += extra_impl[impl_i+1:]

            prem_satur = saturate(prem, other_impl)
            if (prem_satur == cons) or (prem_satur & cons != prem_satur):
                continue
            # if prem_satur is a proper subset of cons
            extra_basis.append((ext, prem_satur, cons))
            extra_elements.append((ext, prem_satur, cons))

        if verbose:
            print('extra elements', [ba_tuple_to_str(elem) for elem in extra_elements])
        return extra_elements

    def add_attribute(attr_idx: int, elements: Tuple[ImplicationType or ConceptType, ...])\
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
        old_stable_impl, new_stable_impl, min_mod_impl, non_min_mod_impl, mod_concepts = [], [], [], [], []
        filter_elements, append_elements = [], []

        if verbose:
            print(f'attr_idx: {attr_idx + 1} begin')
            print('elements:', [ba_tuple_to_str(element) for element in elements])

        for el_idx, element in enumerate(elements, start=1):
            if verbose:
                print(f"iter {attr_idx + 1}.{el_idx}")
                print('element:', ba_tuple_to_str(element))

            to_modify = element[0] & attribute_extents[attr_idx] == element[0]  # element extent is a subset of attr_idx'
            is_concept = len(element) == 2  # element is a concept, i.e. it contains (extent, intent)

            el_to_filter, el_to_append = element, None
            if to_modify and is_concept:
                new_concept, new_min_implication = process_modified_concept(attr_idx, element, min_mod_impl)
                mod_concepts.append(new_concept)
                if new_min_implication:
                    min_mod_impl.append(new_min_implication)
                    el_to_filter = new_min_implication
                else:
                    el_to_filter = None

            elif to_modify and not is_concept:
                new_implication, is_minimal = process_modified_implication(attr_idx, element, min_mod_impl)
                if is_minimal:
                    min_mod_impl.append(new_implication)
                else:
                    non_min_mod_impl.append(new_implication)
                    el_to_filter = None

            elif not to_modify and is_concept:
                new_concept, new_implication = process_stable_concept(attr_idx, element, new_stable_impl)
                if new_concept:
                    el_to_append = new_concept
                if new_implication:
                    el_to_append = new_implication
                    new_stable_impl.append(new_implication)

            elif not to_modify and not is_concept:  # process stable implication
                print('Stable implication') if verbose else None
                old_stable_impl.append(element)
            else:
                raise ValueError("An impossible if-else branch reached")

            filter_elements.append(el_to_filter) if el_to_filter else None
            append_elements.append(el_to_append) if el_to_append else None

            print() if verbose else None

        if verbose:
            for k in [
                'non_min_mod_impl', 'old_stable_impl', 'new_stable_impl', 'min_mod_impl', 'mod_concepts', 'elements'
            ]:
                v = locals()[k]
                if v:
                    print(f"{k}:", [ba_tuple_to_str(element) for element in v])

        fused_elements = fuse(old_stable_impl + new_stable_impl + min_mod_impl, non_min_mod_impl)
        extra_elements = sorted(mod_concepts + fused_elements, key=lambda el: set(ba2iset(el[0])), reverse=True)
        new_elements = filter_elements + append_elements + extra_elements

        if verbose:
            print(f'attr_idx: {attr_idx + 1} end')
            print('=====================')
            print()

        return new_elements

    elements_final: List[ConceptType or ImplicationType] = [(fbarray(~bazeros(N_OBJS)), fbarray(bazeros(N_ATTRS)))]
    for y in tqdm(range(N_ATTRS), desc='Incrementing attributes', disable=not use_tqdm):
        elements_final = add_attribute(y, elements_final)

    return [elem for elem in elements_final if len(elem) == 3]  # return only the implications

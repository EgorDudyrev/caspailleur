from typing import List, Dict, Iterator, Iterable

from .order import topological_sorting
from .base_functions import isets2bas, bas2isets
from .indices import delta_stability_index

from skmine.itemsets import LCM
from bitarray import bitarray, frozenbitarray as fbarray
from bitarray.util import zeros as bazeros
from collections import deque
from tqdm import tqdm


def list_intents_via_LCM(itemsets: List[fbarray], min_supp: float = 1, n_jobs: int = 1) -> List[fbarray]:
    """Get the list of intents by running LCM algorithm from scikit-mine

    Parameters
    ----------
    itemsets:
        The list of itemsets representing the dataset
    min_supp:
        Minimal support for the intent
    n_jobs:
        Number of jobs for LCM algorithm

    Returns
    -------
    intents:
        THe found intents

    """
    n_attrs = len(itemsets[0])

    lcm = LCM(min_supp=min_supp, n_jobs=n_jobs)
    intents = lcm.fit_transform(list(bas2isets(itemsets)))['itemset']
    intents = list(isets2bas(intents, n_attrs))
    intents = topological_sorting(intents)[0]

    biggest_intent = ~bazeros(n_attrs)
    if intents[-1] != biggest_intent:
        intents.append(fbarray(biggest_intent))

    smallest_intent = ~bazeros(n_attrs)
    for itset in itemsets:
        smallest_intent &= itset
        if not smallest_intent.any():
            break
    if intents[0] != smallest_intent:
        intents.insert(0, fbarray(smallest_intent))
    return intents

def list_intents_via_Lindig(itemsets: List[bitarray], attr_extents: List[bitarray]) -> List[List[bitarray]]:
    """Get the list of lists of intents of itemsets grouped by equivalent classes of itemsets running Lindig algorithm
    from "Fast Concept Analysis" by Christian Lindig, Harvard University, Division of Engineering and Applied Sciences

    Parameters
    ----------
    itemsets:
        list of bitarrays representing objects descriptions
    attr_extents:
        list of bitarrays representing extents of attributes

    Returns
    -------
    Lattice_data_intents:
        the list of lists of intents of itemsets grouped by equivalent classes

    """

    class NotFound(Exception):
        pass

    def __down__(intent: List[bitarray], itemsets: List[bitarray]):
        if intent == []:
            return itemsets
        down = intent[0]
        for attr in intent[1:]:
            down = down & attr
        down =  [itemsets[i] for i in range(len(down)) if down[i] == 1]
        return down

    def __up__(extent: List[bitarray], attr_extents: List[bitarray]):
        if extent == []:
            return(attr_extents)
        up = extent[0]
        for obj in extent[1:]:
            up = up & obj
        up = [attr_extents[i] for i in range(len(up)) if up[i] == 1]
        return up

    def check_intersection(list1: List[bitarray], list2: List[bitarray]):
        has_intersection = False

        for bitarray1 in list1:
            for bitarray2 in list2:
                if bitarray1 == bitarray2:
                    has_intersection = True
                    break
        return has_intersection
    
    def compute_extent_bit(extent: List[bitarray], attr_extents: List[bitarray]):
        if extent == []:
            return(bitarray([1 for _ in range(len(attr_extents))]))
        bit_extent = extent[0]
        for obj in extent[1:]:
            bit_extent = bit_extent & obj
        return(bit_extent)
    
    def find_upper_neighbors(concept_extent: List[bitarray], itemsets: List[bitarray], attr_extents: List[bitarray]):
        min_set = [obj for obj in itemsets if obj not in concept_extent]
        neighbors = []

        for g in [obj for obj in itemsets if obj not in concept_extent]:
            B1 = __up__(concept_extent + [g], attr_extents)
            A1 = __down__(B1, itemsets)
            if not check_intersection(min_set, [obj for obj in A1 if obj not in concept_extent and obj not in [g]]):
                neighbors.append(A1)
            else:
                min_set.remove(g)
        return neighbors
    
    def find_next_concept_extent(concept_extent: List[bitarray], List_extents: List[bitarray], attr_extents: List[bitarray]):
        next_concept_extent = None
        for extent in List_extents:
            if compute_extent_bit(extent, attr_extents) < compute_extent_bit(concept_extent, attr_extents) and (next_concept_extent is None or compute_extent_bit(extent, attr_extents) > compute_extent_bit(next_concept_extent, attr_extents)):
                next_concept_extent = extent
        if next_concept_extent is not None:
            return next_concept_extent
        raise NotFound("Next concept not found in Lattice")

        
    Lattice_data_intents = []  # concepts set
    concept_extent = __down__(attr_extents, itemsets)  # Initial Concept
    Lattice_data_intents.append(concept_extent)  # Insert the initial concept into Lattice

    while True:
        for parent in find_upper_neighbors(concept_extent, itemsets, attr_extents):
            if parent not in Lattice_data_intents:
              Lattice_data_intents.append(parent)

        try:
            concept_extent = find_next_concept_extent(concept_extent, Lattice_data_intents)
        except NotFound:
            break
    return Lattice_data_intents

def list_attribute_concepts(intents: List[fbarray]) -> List[int]:
    """Get the indices of `intents` selected by each sole attribute"""
    assert (len(a) <= len(b) for a, b in zip(intents, intents[1:])),\
        'The list of `intents` should be topologically sorted. With the cardinality minimal intent being at the start'

    attr_concepts = [-1] * len(intents[0])
    found_attrs = bazeros(len(intents[0]))
    for intent_i, intent in enumerate(intents):
        for m in intent.itersearch(True):
            if attr_concepts[m] == -1:
                attr_concepts[m] = intent_i
        found_attrs |= intent
        if found_attrs.all():
            break

    return attr_concepts


def iter_equivalence_class(attribute_extents: List[fbarray], intent: fbarray = None) -> Iterator[fbarray]:
    """Iterate subsets of attributes from equivalence class

    The output equivalence class goes from the maximal subsets of attributes to the smallest ones.
    Equivalent subsets of attributes are the ones that describe the same subset of objects.


    Parameters
    ----------
    attribute_extents:
        The list of objects described by each specific attribute (converted to bitarrays)
    intent:
        Intent to compute equivalence class for. If None is passed, Iterate equivalence class of all attributes

    Returns
    -------
    Iterator[frozenbitarray]:
        Iterator over bitarrays representing equivalent subsets of attributes

    """
    N_OBJS, N_ATTRS = len(attribute_extents[0]), len(attribute_extents)

    intent = bazeros(N_ATTRS) if intent is None else intent

    def conjunct_extent(premise: fbarray) -> fbarray:
        res = ~bazeros(N_OBJS)
        for m in premise.itersearch(True):
            res &= attribute_extents[m]
            if not res.any():
                break

        return fbarray(res)

    total_extent = conjunct_extent(intent)
    stack = [[m] for m in intent.itersearch(True)][::-1]

    yield intent
    while stack:
        attrs_to_remove = stack.pop(0)
        last_attr = attrs_to_remove[-1]

        attrs_to_eval = bitarray(intent)
        for m in attrs_to_remove:
            attrs_to_eval[m] = False
        attrs_to_eval = fbarray(attrs_to_eval)

        conj = conjunct_extent(attrs_to_eval)
        if conj != total_extent:
            continue

        # conj == total_extent
        yield attrs_to_eval
        stack += [attrs_to_remove+[m] for m in intent.itersearch(True) if m > last_attr][::-1]


def list_keys_via_eqclass(equiv_class: Iterable[fbarray]) -> List[fbarray]:
    """List minimal subsets from given equivalence class"""
    potent_keys = []
    for new_key in equiv_class:
        potent_keys = [key for key in potent_keys if new_key & key != new_key]
        potent_keys.append(new_key)
    return potent_keys


def list_passkeys_via_eqclass(equiv_class: Iterable[fbarray]) -> List[fbarray]:
    """List subsets of minimal size from given equivalence class"""
    passkeys = []
    for descr in equiv_class:
        if not passkeys or descr.count() == passkeys[-1].count():
            passkeys.append(descr)

        if descr.count() < passkeys[-1].count():
            passkeys = [descr]
    return passkeys


def list_keys(intents: List[fbarray], only_passkeys: bool = False) -> Dict[fbarray, int]:
    """List all keys for all intents (i.e. minimal subsets of attributes selecting specific subsets of objects)"""
    assert all(a.count() <= b.count() for a, b in zip(intents, intents[1:])), \
        'The `intents` list should be topologically sorted by ascending order'

    n_attrs, n_intents = len(intents[-1]), len(intents)
    attrs_descendants = [bazeros(n_intents) for _ in range(n_attrs)]
    for intent_i, intent in enumerate(intents):
        for m in intent.itersearch(True):
            attrs_descendants[m][intent_i] = True

    # assuming that every subset of a key is a key => extending not-a-key cannot result in a key
    # and every subset of a passkey is a passkey

    single_attrs = list(isets2bas([[m] for m in range(n_attrs)], n_attrs))

    if only_passkeys:
        passkey_sizes = [n_attrs] * n_intents

    keys_dict = {fbarray(bazeros(n_attrs)): 0}
    attrs_to_test = deque([m_ba for m_ba in single_attrs])
    while attrs_to_test:
        attrs = attrs_to_test.popleft()
        attrs_indices = list(attrs.itersearch(True))

        if any(attrs & (~single_attrs[m]) not in keys_dict for m in attrs_indices):
            continue

        max_attr_idx = attrs_indices[-1] if attrs_indices else -1
        common_descendants = ~bazeros(n_intents)
        for m in attrs.itersearch(True):
            common_descendants &= attrs_descendants[m]

        meet_intent_idx = common_descendants.find(True)

        if only_passkeys:
            # `attrs` is not a passkey because of the size
            if passkey_sizes[meet_intent_idx] < attrs.count():
                continue

        # if subset of attrs is not a key, or a key of the same intent
        if any(keys_dict[attrs & (~single_attrs[m])] == meet_intent_idx for m in attrs_indices):
            continue

        keys_dict[attrs] = meet_intent_idx
        if only_passkeys:
            passkey_sizes[meet_intent_idx] = attrs.count()
        if meet_intent_idx != n_intents-1:
            attrs_to_test.extend([attrs | m_ba for m_ba in single_attrs[max_attr_idx+1:]])

    return keys_dict


def list_passkeys(intents: List[fbarray]) -> Dict[fbarray, int]:
    """List all passkeys for all intents

     (i.e. subsets of attributes of minimal size selecting specific subsets of objects)"""
    return list_keys(intents, only_passkeys=True)


def list_stable_extents_via_sofia(
        attribute_extents: Iterable[fbarray],
        n_stable_extents: int, min_supp: int = -1,
        use_tqdm: bool = False, n_attributes: int = None
) -> set[fbarray]:
    stable_extents = set()
    for attr_extent in tqdm(attribute_extents, disable=not use_tqdm, total=n_attributes):
        if not stable_extents:  # Set up the top extent, as soon as we know the number of objects
            stable_extents.add(fbarray(~bazeros(len(attr_extent))))

        if attr_extent.count() < min_supp:
            continue

        new_extents = (extent & attr_extent for extent in stable_extents)
        new_extents = filter(lambda extent: extent.count() >= min_supp and extent not in stable_extents, new_extents)
        stable_extents |= set(new_extents)

        if len(stable_extents) > n_stable_extents:
            extents_top_sort = sorted(stable_extents, key=lambda extent: extent.count())
            delta_stabilities = list(delta_stability_index(extents_top_sort))
            stab_thold = sorted(delta_stabilities)[-n_stable_extents]
            stable_extents = {extent for extent, stab in zip(extents_top_sort, delta_stabilities) if stab >= stab_thold}

    if len(stable_extents) > n_stable_extents:
        stable_extents = {extent for extent, stab in zip(extents_top_sort, delta_stabilities) if stab > stab_thold}

    return stable_extents

import heapq
from functools import reduce
from typing import Iterator, Iterable, Union, Sequence, Optional

import deprecation

from .order import topological_sorting, check_topologically_sorted
from .io import isets2bas, bas2isets, to_absolute_number
from .indices import delta_stability_by_extents
from .base_functions import extension

from skmine.itemsets import LCM
from bitarray import bitarray, frozenbitarray as fbarray
from bitarray.util import zeros as bazeros, subset as basubset, count_and
from collections import deque
from tqdm.auto import tqdm


def list_intents_via_LCM(itemsets: list[fbarray], min_supp: Union[int, float] = 0, n_jobs: int = 1) -> list[fbarray]:
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
    min_supp = int(min_supp * len(itemsets) if isinstance(min_supp, float) else min_supp)
    n_attrs = len(itemsets[0])

    lcm = LCM(min_supp=max(min_supp, 1), n_jobs=n_jobs)
    intents = lcm.fit_transform(list(bas2isets(itemsets)))['itemset']
    intents = list(isets2bas(intents, n_attrs))
    intents = topological_sorting(intents)[0]

    biggest_intent_support = sum(itemset.all() for itemset in itemsets)
    if biggest_intent_support >= min_supp:
        if not intents[-1].all():
            intents.append(fbarray(~bazeros(n_attrs)))

    smallest_intent = reduce(bitarray.__and__, itemsets, ~bazeros(n_attrs))
    if intents[0] != smallest_intent:
        intents.insert(0, fbarray(smallest_intent))
    return intents


def list_intents_via_Lindig(itemsets: list[bitarray], attr_extents: list[bitarray]) -> list[bitarray]:
    """Get the list of intents of itemsets grouped by equivalent classes running Lindig algorithm
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
        the list of intents of itemsets grouped by equivalent classes

    """

    class NotFound(Exception):
        pass

    def __down__(intent: list[bitarray], itemsets: list[bitarray]):
        if intent == []:
            return itemsets
        down = intent[0]
        for attr in intent[1:]:
            down = down & attr
        down =  [itemsets[i] for i in range(len(down)) if down[i] == 1]
        return down

    def __up__(extent: list[bitarray], attr_extents: list[bitarray]):
        if extent == []:
            return(attr_extents)
        up = extent[0]
        for obj in extent[1:]:
            up = up & obj
        up = [attr_extents[i] for i in range(len(up)) if up[i] == 1]
        return up

    def check_intersection(list1: list[bitarray], list2: list[bitarray]):
        has_intersection = False
        for bitarray1 in list1:
            for bitarray2 in list2:
                if bitarray1 == bitarray2:
                    has_intersection = True
                    break
        return has_intersection
    
    def compute_extent_bit(extent: list[bitarray], attr_extents: list[bitarray]):
        if extent == []:
            return(bitarray([1 for _ in range(len(attr_extents))]))
        bit_extent = extent[0]
        for obj in extent[1:]:
            bit_extent = bit_extent & obj
        return(bit_extent)
    
    def find_upper_neighbors(concept_extent: list[bitarray], itemsets: list[bitarray], attr_extents: list[bitarray]):
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
    
    def find_next_concept_extent(concept_extent: list[bitarray], List_extents: list[bitarray], attr_extents: list[bitarray]):
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
            concept_extent = find_next_concept_extent(concept_extent, Lattice_data_intents, attr_extents)
        except NotFound:
            break
    for i in range(len(Lattice_data_intents)):
      Lattice_data_intents[i] = compute_extent_bit(Lattice_data_intents[i], attr_extents)
        
    return Lattice_data_intents


def list_attribute_concepts(intents: list[fbarray]) -> list[int]:
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


def iter_equivalence_class(attribute_extents: list[fbarray], intent: fbarray = None, presort_output = True) -> Iterator[fbarray]:
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

    intent = set(range(N_ATTRS)) if intent is None else set(intent.search(True))

    total_extent = extension(intent, attribute_extents)

    antigenerator, next_antigenerators = None, deque([tuple()])  # antigenerator: s.t. ext(intent\antigenerator) = ext(intent)
    for level in range(0, len(intent) + 1):
        antigenerators, next_antigenerators = next_antigenerators, deque()
        antigenerators = list(antigenerators)
        for antigenerator in antigenerators:
            generator = intent - set(antigenerator)

            extent = extension(generator, attribute_extents)
            if extent == total_extent:
                yield next(isets2bas([generator], N_ATTRS))
                next_antigenerators.append(antigenerator)

        if not next_antigenerators:
            break

        next_antigenerators = [antigen for antigen, _ in
                               generate_next_level_descriptions(next_antigenerators, n_attributes=N_ATTRS)]
        if presort_output:
            next_antigenerators = sorted(next_antigenerators, reverse=True)


def list_keys_via_eqclass(equiv_class: Iterable[fbarray]) -> list[fbarray]:
    """List minimal subsets from given equivalence class"""
    potent_keys = []
    for new_key in equiv_class:
        potent_keys = [key for key in potent_keys if new_key & key != new_key]
        potent_keys.append(new_key)
    return potent_keys


def list_passkeys_via_eqclass(equiv_class: Iterable[fbarray]) -> list[fbarray]:
    """List subsets of minimal size from given equivalence class"""
    passkeys = []
    for descr in equiv_class:
        if not passkeys or descr.count() == passkeys[-1].count():
            passkeys.append(descr)

        if descr.count() < passkeys[-1].count():
            passkeys = [descr]
    return passkeys


def iter_keys_of_intent(intent: fbarray, attr_extents: list[fbarray]) -> Iterator[fbarray]:
    n_attrs = len(intent)
    single_attrs = list(isets2bas([{attr_idx} for attr_idx in range(n_attrs)], n_attrs))
    extent = extension(intent, attr_extents)

    def subdescriptions(description: fbarray) -> Iterator[fbarray]:
        return (description & ~single_attrs[m_i] for m_i in description.itersearch(True))

    key_candidates = deque([intent])
    while key_candidates:
        key_candidate = key_candidates.popleft()
        equiv_subdescrs = [descr for descr in subdescriptions(key_candidate)
                           if extension(descr, attr_extents) == extent]
        if not equiv_subdescrs:
            yield key_candidate
            continue

        key_candidates.extend(equiv_subdescrs)


def iter_keys_of_intent_pretentious(intent: fbarray, attr_extents: list[fbarray]) -> list[fbarray]:
    # TODO: Test the function and check if it works right and faster than the straightforward solution
    n_attrs = len(intent)
    intent = set(intent.search(True))
    extent = extension(intent, attr_extents)

    # The cycle structure is inspired by Talky-G algorithm and reverse pre-order traversal by L. Szathmary et al.
    key_candidates = [intent]
    sub_descrs_to_remove = deque([{attr_idx} for attr_idx in sorted(intent)
                                  if extension(intent-{attr_idx}, attr_extents) == extent])
    while sub_descrs_to_remove:
        attrs_to_remove = sub_descrs_to_remove.pop()
        key_candidate = intent - attrs_to_remove

        rightmost_attr = max(attrs_to_remove) if attrs_to_remove else -1
        attrs_to_remove_next = [attr_idx for attr_idx in intent
                                if rightmost_attr < attr_idx
                                and extension(key_candidate - {attr_idx}, attr_extents) == extent]
        if not attrs_to_remove_next:
            key_candidates.append(key_candidate)
            continue

        sub_descrs_to_remove.extend([attrs_to_remove | {attr_idx} for attr_idx in attrs_to_remove_next])

    single_attrs = list(isets2bas([{attr_idx} for attr_idx in range(n_attrs)], n_attrs))
    key_candidates = [reduce(fbarray.__or__, map(single_attrs.__getitem__, candidate), fbarray(bazeros(n_attrs)))
                      for candidate in key_candidates]
    key_candidates = deque(sorted(key_candidates, key=lambda candidate: candidate.count()))
    for i in range(len(key_candidates)):
        if i >= len(key_candidates):
            break

        key = key_candidates[i]
        j = i+1
        while j < len(key_candidates):
            if basubset(key, key_candidates[j]):
                del key_candidates[j]
                continue
            j += 1

    return list(key_candidates)


def list_keys(intents: list[fbarray], only_passkeys: bool = False) -> dict[fbarray, int]:
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

        common_descendants = ~bazeros(n_intents)
        for m in attrs.itersearch(True):
            common_descendants &= attrs_descendants[m]
        if not common_descendants.any():
            continue
        meet_intent_idx = common_descendants.index(True)

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
        if not intents[meet_intent_idx].all():
            max_attr_idx = attrs_indices[-1] if attrs_indices else -1
            attrs_to_test.extend([attrs | m_ba for m_ba in single_attrs[max_attr_idx+1:]])

    return keys_dict


def list_passkeys(intents: list[fbarray]) -> dict[fbarray, int]:
    """List all passkeys for all intents

     (i.e. subsets of attributes of minimal size selecting specific subsets of objects)"""
    return list_keys(intents, only_passkeys=True)


def list_keys_for_extents(
        extents: list[fbarray], attr_extents: list[fbarray],
        only_passkeys: bool = False
) -> dict[fbarray, int]:
    """List all keys for given extents (i.e. minimal subsets of attributes selecting given closed subsets of objects)"""
    if not check_topologically_sorted(extents, ascending=False):
        extents, orig_to_topsort_idx_map = topological_sorting(extents, ascending=False)
    else:
        orig_to_topsort_idx_map = list(range(len(extents)))
    topsort_to_orig_idx_map = {sort_idx: orig_idx for orig_idx, sort_idx in enumerate(orig_to_topsort_idx_map)}

    extents_idx_map: dict[fbarray, int] = {extent: extent_i for extent_i, extent in enumerate(extents)}
    min_support = extents[-1].count()

    n_attrs, n_objs, n_extents = len(attr_extents), len(attr_extents[0]), len(extents)
    total_extent = attr_extents[0] | ~attr_extents[0]

    single_attrs = list(isets2bas([[m] for m in range(n_attrs)], n_attrs))

    if only_passkeys:
        passkey_sizes = [n_attrs] * n_extents

    keys_dict: dict[fbarray, tuple[int, int|None]] = {fbarray(bazeros(n_attrs)): (n_objs, extents_idx_map.get(total_extent))}
    testing_stack = deque(single_attrs)  # iteration order is inspired by Talky-G algorithm
    while testing_stack:
        attrs = testing_stack.pop()
        attrs_indices = list(attrs.itersearch(True))
        sub_descriptions = [attrs & (~single_attrs[m]) for m in attrs_indices]

        # check that every subset of attrs is a key
        if any(subattrs not in keys_dict for subattrs in sub_descriptions):
            continue

        curr_extent: fbarray = reduce(fbarray.__and__, map(attr_extents.__getitem__, attrs_indices), total_extent)
        curr_extent_i = extents_idx_map[curr_extent] if curr_extent in extents_idx_map else None

        if only_passkeys and curr_extent_i is not None:
            # `attrs` is not a passkey because of the size
            if passkey_sizes[curr_extent_i] < attrs.count():
                continue

        # if subset of attrs has the same support then it is not a key
        support = curr_extent.count()
        if any(keys_dict[subattrs][0] <= support for subattrs in sub_descriptions):
            continue

        keys_dict[attrs] = (support, curr_extent_i)
        if only_passkeys and curr_extent_i is not None:
            passkey_sizes[curr_extent_i] = attrs.count()

        has_smaller_extents = support > min_support
        if curr_extent_i is not None and has_smaller_extents:
            has_smaller_extents = any(basubset(ext, curr_extent) for ext in extents[curr_extent_i+1:])

        if has_smaller_extents:
            max_attr_idx = attrs_indices[-1] if attrs_indices else -1
            testing_stack.extend([attrs | m_ba for m_ba in single_attrs[max_attr_idx+1:]])

    return {key: topsort_to_orig_idx_map[extent_i]
            for key, (support, extent_i) in keys_dict.items() if extent_i is not None}


def list_passkeys_for_extents(
        extents: list[fbarray], attr_extents: list[fbarray],
) -> dict[fbarray, int]:
    return list_keys_for_extents(extents, attr_extents, only_passkeys=True)


@deprecation.deprecated(deprecated_in="0.2.0", removed_in="0.2.1",
                        details="Use `list_stable_extents_via_gsofia` function instead. It is faster and more reliable")
def list_stable_extents_via_sofia(
        attribute_extents: Iterable[fbarray],
        n_stable_extents: int, min_supp: int = 0,
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
            delta_stabilities = list(delta_stability_by_extents(extents_top_sort))
            stab_thold = sorted(delta_stabilities)[-n_stable_extents]
            stable_extents = {extent for extent, stab in zip(extents_top_sort, delta_stabilities) if stab >= stab_thold}

    if len(stable_extents) > n_stable_extents:
        stable_extents = {extent for extent, stab in zip(extents_top_sort, delta_stabilities) if stab > stab_thold}

    return stable_extents


def list_stable_extents_via_gsofia(
        attribute_extents: Iterable[fbarray], n_objects: int = None,
        min_delta_stability: Union[int, float] = 0,
        n_stable_extents: int = None,
        min_supp: Union[int, float] = 0,
        use_tqdm: bool = False, n_attributes: int = None,
) -> set[fbarray]:
    """Select the most stable extents restricted by min. delta stability value, n. most stable extents, and min. support

    An extent is ∆-stable if there exists no sub-extent that covers ∆ fewer objects than the given extent.

    Extent is a subset of objects, described by some attributes.
    Every extent is an intersection of a subset of `attribute_extents`.


    Parameters
    ----------
    attribute_extents:
        Iterable over extents of attributes.
        Every extent is a set of objects, described by an attribute represented with a bitarray.
    n_objects:
        The number of objects in the data.
        Have to be specified unless it can be computed as `len(attribute_extents[0])`.
    min_delta_stability:
        A lower bound for delta stability of the returned extents.
        The parameter should represent either an absolute number or a percentage of objects in the data.
        Delta stability shows how many objects a description will lose if made a bit more precise.
        Delta-stable extents are the ones that are selected by delta-stable descriptions.
    n_stable_extents:
        An upper bound on the number of the returned extents.
        The bigger is the number, the longer it takes to run the algorithm and the more precise are the results.
    min_supp:
        A lower bound on the support of the returned extents.
        Support of the extent shows how many objects are contained in this extent.
    use_tqdm:
        A flag, whether to visualise the progress bar via tqdm package.
        If set to True and `attribute_extents` is a generator,
         please specify `n_attributes` parameter to define the maximal amount of steps of the algorithm.
    n_attributes:
        The number of elements in `attribute_extents` iterator.
        Optional parameter that one might need to specify in order to set up the width of tqdm progress bar
        (if `use_tqdm` is set to True, and the length of `attribute_extents` cannot be computed automatically).


    Returns
    -------
    set[frozenbitarray]:
        A set of stable extents computed w.r.t. the other parameters.

    Notes
    -----
    The algorithm falls in line with gSofia algorithm presented in
    paper "Efficient Mining of Subsample-Stable Graph Patterns"
    by A. Buzmakov, S.O. Kuznetsov, and A. Napoli from International Conference on Data Mining of 2017.
    However, we remove the graph-specific part of the algorithm to make it applicable for any type of attributes.

    """
    if n_objects is None:
        try:
            n_objects = len(attribute_extents[0])
        except TypeError:
            raise TypeError('The function cannot establish the number of objects. '
                            'Please, either provide the number of objects with `n_objects` parameter '
                            'or convert `attribute_extents` parameter into a list')
    if n_attributes is None:
        try:
            n_attributes = len(attribute_extents)
        except:
            pass
    min_delta_stability = to_absolute_number(min_delta_stability, n_objects)
    min_supp = to_absolute_number(min_supp, n_objects)

    # Create dict with a format:  extent => (delta_index, children_extents)
    top_extent = fbarray(~bazeros(n_objects))
    stable_extents: dict[fbarray, tuple[int, set[fbarray]]] = {top_extent: (n_objects, set())}

    # noinspection PyTypeChecker
    attr_extent_iterator: Iterable[fbarray] = tqdm(attribute_extents, total=n_attributes, disable=not use_tqdm)
    for attr_extent in attr_extent_iterator:
        # The following code mimics "ExtendProjection" function from gSofia algorithm
        old_stable_extents, stable_extents = dict(stable_extents), dict()
        for extent, (delta, children) in old_stable_extents.items():
            # Create new extent
            extent_new = extent & attr_extent
            if extent_new == extent:
                stable_extents[extent] = (delta, children)
                continue

            # Update the stability of the old extent given its new child: `extent_new`
            delta = min(delta, extent.count() - extent_new.count())
            if delta >= min_delta_stability:
                stable_extents[extent] = (delta, children | {extent_new})

            # Skip the new extent if it is too small
            if extent_new.count() < min_supp:
                continue

            # Find the delta-index of the new extent and its children extents
            delta_new, children_new = extent_new.count(), []
            for child in children:
                child_new = child & attr_extent
                delta_new = min(delta_new, extent_new.count() - child_new.count())
                if delta_new < min_delta_stability:
                    children_new = []
                    break
                children_new.append(child_new)

            # Skip the new extent if it is too unstable
            if delta_new < min_delta_stability:
                continue

            # Filter the maximal children (so filter out transitive children that are children to other children)
            children_new, i = sorted(children_new, key=lambda ext: ext.count(), reverse=True), 0
            while i < len(children_new):
                child = children_new[i]
                has_bigger_child = any(basubset(child, bigger_child) for bigger_child in children_new[:i])
                if has_bigger_child:
                    del children_new[i]
                else:
                    i += 1
            children_new = set(children_new)

            # At this point, we know that `extent_new` is big enough and is stable enough
            stable_extents[extent_new] = (delta_new, children_new)

        # If only the best stable concepts required
        if n_stable_extents is not None and len(stable_extents) > n_stable_extents:
            most_stable_extents = heapq.nlargest(n_stable_extents, stable_extents.items(), key=lambda x: x[1][0])
            thold = most_stable_extents[-1][1][0]
            n_border_stable = 0
            for (_, (stab, _)) in most_stable_extents[::-1]:
                if stab > thold:
                    continue
                n_border_stable += 1
            n_border_stable_total = sum(stab == thold for _, (stab, _) in stable_extents.items())
            if n_border_stable < n_border_stable_total:
                most_stable_extents = most_stable_extents[:-n_border_stable]
            stable_extents = dict(most_stable_extents)

    return set(stable_extents)


def generate_next_level_descriptions(
        same_level_descriptions: Sequence[tuple[int, ...]],
        attribute_extents: Sequence[fbarray] = None,
        n_attributes: int = None
) -> Iterator[tuple[tuple[int, ...], Optional[int]]]:
    """Generate the next level descriptions from the given ones

    Descriptions (i.e. set of attributes/items) belong to the same level when they have the same length.
    Description of n+1 attributes will only be generated
    only if all its subdescriptions of n attributes can be found in `current_level_descriptions`.

    Parameters
    ----------
    same_level_descriptions:
        Sequence of descriptions (as tuples of indices of their attributes) of the same length
    attribute_extents:
        Sequence extents of attributes.
        Every extent is a set of objects described by an attribute and represented with a bitarray.
        The parameter is optional, and it provides a slight optimisation
        for support computations of generated descriptions
    n_attributes:
        Number of attributes. The parameter is only required when no `attribute_extents` are provided
        and the `same_level_descriptions` contain one empty-set description.

    Returns
    -------
    Iterator of pairs (next_level_description, next_level_description_support) where
    next_level_description: tuple[int, ...]
        Next-level description (as a tuple of indices of its attributes) composed of the given
        `current_level_descriptions`.
    next_level_description_support: int | None
        The support of the corresponding next_level_description. (if `attribute_extents` is provided, else None)
        Support of a description is the number of objects it describes.

    """
    provide_support = attribute_extents is not None
    zero_level = next(len(descr) for descr in same_level_descriptions) == 0

    n_attributes = len(attribute_extents) if attribute_extents is not None else n_attributes
    if n_attributes is None:
        if not zero_level:
            n_attributes = max(max(descr) for descr in same_level_descriptions) + 1
        else:
            raise ValueError('Provide `n_attributes` parameter to `generate_next_level_descriptions` functions. '
                             'As it is not deducible from the values of the other parameters.')

    if zero_level:
        for next_attr in range(n_attributes):
            yield (next_attr,), attribute_extents[next_attr].count() if provide_support else None
        return

    possible_suffixes: dict[tuple[int, ...], list[int]] = {}
    for description in same_level_descriptions:
        if description[:-1] not in possible_suffixes:
            possible_suffixes[description[:-1]] = []
        possible_suffixes[description[:-1]].append(description[-1])
    possible_suffixes = {description: set(suffixes) for description, suffixes in possible_suffixes.items()}

    for description in same_level_descriptions:
        subdescriptions = [description[:i]+description[i+1:] for i in range(len(description))]
        if any(subdescription not in possible_suffixes for subdescription in subdescriptions):
            continue

        extent = extension(description, attribute_extents) if provide_support else None
        next_attributes = reduce(set.intersection, (possible_suffixes[subgen] for subgen in subdescriptions))
        for next_attr in next_attributes:
            if next_attr <= description[-1]:
                continue
            next_support = count_and(extent, attribute_extents[next_attr]) if provide_support else None
            yield description + (next_attr, ), next_support


def iter_minimal_rare_itemsets_via_mrgexp(
        attribute_extents: list[fbarray], max_support: int,
        max_length: int = None
) -> Iterator[fbarray]:
    """List minimal rare itemsets using MRG-Exp (aka Carpathia-G-Rare) algorithm

    A minimal rare itemset (or a minimal rare description) is a minimal subset of attributes
    that describes less than (or equal to) `max_support` objects.
    Minimality here means that any subset of a minimal rare itemset describes more than `max_support` objects.

    Parameters
    ----------
    attribute_extents:
        Sequence extents of attributes.
        Every extent is a set of objects described by an attribute and represented with a bitarray.
    max_support:
        Maximal number of objects that should be described by an itemset (aka a description).
    max_length:
        Maximum size of a rare itemset.
        Default value: the number of attributes: len(attribute_extents).

    Returns
    -------
    minimal_rare_itemsets:
        Minimal rare itemsets found by the algorithm.
        The itemsets are placed in the order of increasing sizes:
        the first itemset contains the fewer attributes, the latter contains the maximal number of attributes.

    Notes
    -----

    The algorithm is introduced in Szathmary, L., Napoli, A., & Valtchev, P. (2007, October). Towards rare itemset mining.
    In 19th IEEE international conference on tools with artificial intelligence (ICTAI 2007) (Vol. 1, pp. 305-312). IEEE.
    """
    n_attrs = len(attribute_extents)
    max_length = n_attrs if max_length is None else max_length
    total_extent = attribute_extents[0] | ~attribute_extents[0]

    prev_level_generators, cur_level_gens = None, {tuple(): total_extent.count()}
    for level in range(1, max_length + 1):
        prev_level_generators, cur_level_gens = cur_level_gens, {}
        if not prev_level_generators:
            break

        new_candidates = generate_next_level_descriptions(prev_level_generators, attribute_extents)
        for new_generator, new_support in new_candidates:
            sub_generators = (new_generator[:i] + new_generator[i + 1:] for i in range(level))

            not_a_generator = any(
                sub_gen not in prev_level_generators or new_support == prev_level_generators[sub_gen]
                for sub_gen in sub_generators
            )
            if not_a_generator:
                continue

            if new_support <= max_support:
                yield next(isets2bas([new_generator], n_attrs))
                continue

            cur_level_gens[new_generator] = new_support


def iter_minimal_broad_clusterings_via_mrgexp(
        attribute_extents: list[fbarray], min_coverage: int,
        max_length: int = None,
        min_added_coverage: int = 1
) -> Iterator[fbarray]:
    """Iterate minimal broad clusterings using an analogue of MRG-Exp algorithm for minimal rare itemsets mining

    A minimal broad clustering is a minimal subset of attributes
    that, together, cover more than (or equal to) `min_coverage` objects.
    Minimality here means that any subset of a minimal broad clustering describes less than `min_coverage` objects.

    Coverage of a clustering is the number of objects lying in the union of all the clusters
    (here 'a cluster' is a synonym of 'an attribute').

    Parameters
    ----------
    attribute_extents:
        Sequence extents of attributes.
        Every extent is a set of objects described by an attribute and represented with a bitarray.
    min_coverage:
        Minimal number of objects that should be covered by all the clusters (attributes) together
    max_length:
        Maximum size of a clustering.
        Default value: the number of attributes: len(attribute_extents).
    min_added_coverage:
        Minimal number of objects that a cluster (i.e. an attribute) should bring to a clustering.
        For example, for a clustering {a, b, c}, its every subset ({a,b}, {a, c}, {b, c}) should cover
        less than `coverage({a,b,c}) - min_added_coverage` objects.

    Returns
    -------
    minimal_broad_clusterings:
        Minimal broad clusterings found by the algorithm.
        The clusterings are placed in the order of increasing sizes:
        the first clustering contains the fewer attributes, the latter contains the maximal number of attributes.

    Notes
    -----

    The algorithm was introduced in:
    E.Dudyrev et al. "Clustering with Stable Pattern Concepts"
    Published in Amedeo Napoli and Sebastian Rudolph (Eds.):
    The 12th International Workshop "What can FCA do for Artificial Intelligence?",
    FCA4AI 2024, co-located with ECAI 2024, October 19 2024, Santiago de Compostela, Spain.
    """
    n_objs, n_attrs = len(attribute_extents[0]), len(attribute_extents)
    max_length = n_attrs if max_length is None else max_length
    empty_extent = attribute_extents[0] & ~attribute_extents[0]
    leftovers = [~extent for extent in attribute_extents]

    prev_level_generators, cur_level_gens = None, {tuple(): empty_extent.count()}
    for level in range(1, max_length + 1):
        prev_level_generators, cur_level_gens = cur_level_gens, {}
        if not prev_level_generators:
            break

        new_candidates = generate_next_level_descriptions(prev_level_generators, leftovers)
        for new_generator, new_leftovers_support in new_candidates:
            new_coverage = n_objs - new_leftovers_support
            sub_generators = (new_generator[:i] + new_generator[i + 1:] for i in range(level))

            not_a_generator = any(
                sub_gen not in prev_level_generators
                or new_coverage < prev_level_generators[sub_gen] + min_added_coverage
                for sub_gen in sub_generators
            )
            if not_a_generator:
                continue

            if new_coverage >= min_coverage:
                yield next(isets2bas([new_generator], n_attrs))
                continue

            cur_level_gens[new_generator] = new_coverage

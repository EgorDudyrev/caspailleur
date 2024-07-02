"""Module with easy to use general functions for working with Caspailleur"""
from operator import itemgetter
import typing
from typing import Iterator, Iterable, Literal, Union, Optional, get_args
import pandas as pd
from bitarray import frozenbitarray as fbarray

from .base_functions import powerset, extension, intention
from .io import ContextType, to_bitarrays, to_itemsets, transpose_context, isets2bas, verbalise, to_absolute_number
from .order import topological_sorting
from . import indices as idxs
from . import definitions
from . import mine_equivalence_classes as mec, implication_bases as ibases


def iter_descriptions(data: ContextType) -> Iterator[dict]:
    bitarrays, objects, attributes = to_bitarrays(data)
    attr_extents = transpose_context(bitarrays)

    sub_pseudo_intents = []

    for description_idxs in powerset(range(len(attributes))):
        extent_ba = fbarray(extension(description_idxs, attr_extents))
        stats = {
            'description': set(verbalise(description_idxs, attributes)),
            'extent': set(verbalise(extent_ba, objects)),
            'intent': set(verbalise(definitions.closure(description_idxs, attr_extents), attributes)),
            'support': idxs.support_by_description(description_idxs, attr_extents, extent_ba),
            'delta-stability': idxs.delta_stability_by_description(description_idxs, attr_extents),
            'is_closed': definitions.is_closed(description_idxs, attr_extents),
            'is_key': definitions.is_key(description_idxs, attr_extents),
            'is_passkey': definitions.is_passkey(description_idxs, attr_extents),
            'is_proper_premise': definitions.is_proper_premise(description_idxs, attr_extents),
            'is_pseudo_intent': definitions.is_pseudo_intent(description_idxs, attr_extents, sub_pseudo_intents)
        }
        yield stats
        if stats['is_pseudo_intent']:
            sub_pseudo_intents.append(description_idxs)


def mine_descriptions(
        data: ContextType,
        min_support: Union[int, float] = 0,
) -> pd.DataFrame:
    """Mine all frequent descriptions and their characteristics


    Note: If you want to look at only the most stable descriptions, then use "mine_concepts" function,
     as all stable descriptions are concept intents.
    """
    bitarrays, objects, attributes = to_bitarrays(data)

    n_objects = len(objects)
    min_support = to_absolute_number(min_support, n_objects)

    attr_extents = transpose_context(bitarrays)

    intents_ba = mec.list_intents_via_LCM(bitarrays, min_supp=min_support)
    extents_ba = [fbarray(extension(intent, attr_extents)) for intent in intents_ba]
    keys_ba = mec.list_keys(intents_ba)
    passkeys_ba = mec.list_passkeys(intents_ba)
    ppremises_ba = dict(ibases.iter_proper_premises_via_keys(intents_ba, keys_ba))
    pintents_ba = dict(ibases.list_pseudo_intents_via_keys(ppremises_ba.items(), intents_ba))

    ext_int_map = dict(zip(extents_ba, intents_ba))

    descriptions = list(map(set, powerset(range(len(attributes)))))
    descr_df = pd.DataFrame(index=pd.Series(descriptions, name='description_idxs'))
    descr_df['description'] = [set(verbalise(descr_idxs, attributes)) for descr_idxs in descr_df.index]
    descr_df['extent_ba'] = [extension(descr, attr_extents) for descr in descr_df.index]
    descr_df['support'] = [idxs.support_by_description(..., ..., extent) for extent in descr_df['extent_ba']]
    descr_df = descr_df[descr_df['support'] >= min_support]

    descr_df['extent'] = [set(verbalise(extent_ba, objects)) for extent_ba in descr_df['extent_ba']]
    descr_df['intent'] = [set(verbalise(ext_int_map[extent_ba], attributes)) for extent_ba in descr_df['extent_ba']]
    descr_df['delta-stability'] = [idxs.delta_stability_by_description(descr, attr_extents, extent)
                                   for descr, extent in descr_df['extent_ba'].items()]
    descr_df['is_closed'] = descr_df['intent'] == descr_df['description']
    descr_df['descr_ba'] = list(isets2bas(descr_df.index, len(attributes)))


    descr_df['is_key'] = [descr_ba in keys_ba for descr_ba in descr_df['descr_ba']]
    descr_df['is_passkey'] = [descr_ba in passkeys_ba for descr_ba in descr_df['descr_ba']]

    descr_df['is_proper_premise'] = [descr_ba in ppremises_ba for descr_ba in descr_df['descr_ba']]
    descr_df['is_pseudo_intent'] = [descr_ba in pintents_ba for descr_ba in descr_df['descr_ba']]

    cols_order = [
        'description', 'extent', 'intent',
        'support', 'delta-stability',
        'is_closed', 'is_key', 'is_passkey', 'is_proper_premise', 'is_pseudo_intent'
    ]

    descr_df = descr_df.reset_index(drop=True).reindex(columns=cols_order).dropna(axis='columns')
    return descr_df


MINE_CONCEPTS_COLUMN = Literal[
    "extent", "intent", "support", "delta-stability",
    "keys", "passkeys", "proper_premises", "pseudo_intents"
]


def mine_concepts(
        data: ContextType,
        to_compute: Optional[Union[list[MINE_CONCEPTS_COLUMN], Literal['all']]] = (
                'extent', 'intent', 'support', 'delta-stability', 'keys', 'passkeys', 'proper_premises'),
        min_support: Union[int, float] = 0,
        min_delta_stability: Union[int, float] = 0, n_stable_concepts: Optional[int] = None,
        use_tqdm: bool = False,
        return_all_computed: bool = False
) -> pd.DataFrame:
    ##################################################
    # Computing what columns and parameters to compute
    ##################################################
    all_cols = list(typing.get_args(MINE_CONCEPTS_COLUMN))
    cols_to_return = list(to_compute) if to_compute is not None and to_compute != 'all' else list(all_cols)
    cols_to_compute = set(cols_to_return)
    assert set(cols_to_compute) <= set(all_cols), \
        f"You asked `to_compute` columns {cols_to_compute}. " \
        f"However, only the following columns can be chosen: {MINE_CONCEPTS_COLUMN}."

    for step, dependencies in [
        ('pseudo_intents', {'proper_premises', 'intent'}),
        ('proper_premises', {'intent', 'keys'}),
        ('passkeys', {'intent'}),
        ('keys', {'intent'}),
        ('delta-stability', {'intent', 'extent'}),
        ('support', {'extent'}),
        ('extent', {'intent'})
    ]:
        cols_to_compute.update(dependencies if step in cols_to_compute else [])

    # whether to compute all concepts whose support is higher min_support
    compute_join_semilattice = min_delta_stability == 0 and n_stable_concepts is None
    if compute_join_semilattice and ('keys' in to_compute or 'passkeys' in to_compute):
        cols_to_compute.add('extent')

    if return_all_computed:
        cols_to_return += sorted(set(cols_to_compute) - cols_to_compute, key=all_cols.index)

    #################################
    # Running the (long) computations
    #################################
    bitarrays, objects, attributes = to_bitarrays(data)
    attr_extents = transpose_context(bitarrays)

    if 'intent' in cols_to_compute:
        if compute_join_semilattice:
            intents_ba = mec.list_intents_via_LCM(bitarrays, min_supp=min_support)
        else:
            n_objects = len(objects)
            stable_extents = mec.list_stable_extents_via_gsofia(
                attr_extents, n_objects, min_delta_stability, n_stable_concepts,
                min_supp=to_absolute_number(min_support, n_objects), n_attributes=len(attr_extents)
            )
            intents_ba = [intention(extent, attr_extents) for extent in stable_extents]

        intents_ba = topological_sorting(intents_ba)[0]
        n_concepts = len(intents_ba)
    if 'extent' in cols_to_compute:
        extents_ba = [fbarray(extension(intent, attr_extents)) for intent in intents_ba]
    if 'keys' in cols_to_compute:
        if compute_join_semilattice:
            keys_ba = mec.list_keys(intents_ba)
        else:
            keys_ba = mec.list_keys_for_extents(extents_ba, attr_extents)
    if 'passkeys' in cols_to_compute:
        if compute_join_semilattice:
            passkeys_ba = mec.list_passkeys(intents_ba)
        else:
            passkeys_ba = mec.list_passkeys_for_extents(extents_ba, attr_extents)
    if 'proper_premises' in cols_to_compute:
        ppremises_ba = dict(ibases.iter_proper_premises_via_keys(intents_ba, keys_ba))
    if 'pseudo_intents' in cols_to_compute:
        pintents_ba = dict(ibases.list_pseudo_intents_via_keys(
            ppremises_ba.items(), intents_ba, use_tqdm=use_tqdm, n_keys=len(ppremises_ba)))
    if 'delta-stability' in cols_to_compute:
        delta_stabilities = [idxs.delta_stability_by_description(descr, attr_extents, extent_ba)
                             for descr, extent_ba in zip(intents_ba, extents_ba)]

    ##############################################
    # Put all the computed values into a DataFrame
    ##############################################
    def get_vals_for_column(column_name):
        def group_by_concept(pairs: Iterable[tuple[fbarray, int]], n_cncpts: int) -> list[list[set[str]]]:
            per_concept = [[] for _ in range(n_cncpts)]
            for ba, cncpt_i in pairs:
                per_concept[cncpt_i].append(ba)
            return per_concept

        def verbalise_descriptions(bas):
            return [verbalise(ba, attributes) for ba in bas]

        if column_name == 'intent':
            return verbalise_descriptions(intents_ba)
        if column_name == 'extent':
            return [verbalise(ba, objects) for ba in extents_ba]
        if column_name == 'support':
            return [extent.count() for extent in extents_ba]
        if column_name == 'delta-stability':
            return delta_stabilities
        if column_name == 'keys':
            return [verbalise_descriptions(keys) for keys in group_by_concept(keys_ba.items(), n_concepts)]
        if column_name == 'passkeys':
            return [verbalise_descriptions(pkeys) for pkeys in group_by_concept(passkeys_ba.items(), n_concepts)]
        if column_name == 'proper_premises':
            return [verbalise_descriptions(pps) for pps in group_by_concept(ppremises_ba.items(), n_concepts)]
        if column_name == 'pseudo_intents':
            return [verbalise_descriptions(pis) for pis in group_by_concept(pintents_ba.items(), n_concepts)]
        raise NotImplementedError('Found some problem with the code')

    concepts_df = pd.DataFrame({f: get_vals_for_column(f) for f in cols_to_return})
    return concepts_df


BASIS_NAME = Literal[
    "Proper Premise", "Canonical Direct", "Karell",
    "Pseudo-Intent", "Canonical", "Duquenne-Guigues", "Minimum",
]

MINE_IMPLICATIONS_COLUMN = Literal[
    "premise", "conclusion", "conclusion_full", "extent", "support"
]


def mine_implications(
        data: ContextType, basis_name: BASIS_NAME = 'Proper Premise',
        unit_base: bool = False,
        to_compute: Optional[Union[list[MINE_IMPLICATIONS_COLUMN], Literal['all']]] = 'all',
) -> pd.DataFrame:
    assert basis_name in get_args(BASIS_NAME), \
        f"You asked for '{basis_name}' basis. But only the following bases are supported: {BASIS_NAME}"
    if basis_name in {'Canonical Direct', "Karell"}:
        basis_name = 'Proper Premise'
    if basis_name in {'Canonical', 'Duquenne-Guigues', 'Minimum'}:
        basis_name = 'Pseudo-Intent'

    to_compute = list(get_args(MINE_IMPLICATIONS_COLUMN)) if to_compute is None or to_compute == 'all' else to_compute
    assert set(to_compute) <= set(get_args(MINE_IMPLICATIONS_COLUMN)), \
        f"You asked `to_compute` columns {to_compute}. " \
        f"However, only the following columns can be chosen: {MINE_IMPLICATIONS_COLUMN}."

    bitarrays, objects, attributes = to_bitarrays(data)
    attr_extents = transpose_context(bitarrays)

    intents_ba = mec.list_intents_via_LCM(bitarrays)
    keys_ba = mec.list_keys(intents_ba)
    ppremises_ba = list(ibases.iter_proper_premises_via_keys(intents_ba, keys_ba))
    basis = ppremises_ba
    if basis_name == 'Pseudo-Intent':
        basis = ibases.list_pseudo_intents_via_keys(ppremises_ba, intents_ba)

    # compute pseudo-closures to reduce the conclusions by attributes implied by other implications
    if 'conclusion' in to_compute:
        pseudo_closures = [
            ibases.saturate(premise, basis[:impl_i] + basis[impl_i + 1:], intents_ba)
            for impl_i, (premise, intent_i) in enumerate(basis)
        ]
        basis = [(premise, intents_ba[intent_i] & ~pintent, intent_i)
                 for (premise, intent_i), pintent in zip(basis, pseudo_closures)]
    else:
        basis = [(premise, None, intent_i) for premise, intent_i in basis]

    if unit_base and 'conclusion' in to_compute:
        single_attrs = to_bitarrays([{i} for i in range(len(attributes))])[0]
        basis = [(premise, single_attrs[attr_i], intent_i)
                 for premise, conclusion, intent_i in basis for attr_i in conclusion.search(True)]

    premises, conclusions, intents_idxs = zip(*basis)

    if "extent" in to_compute or "support" in to_compute:
        extents_ba = [fbarray(extension(intent, attr_extents)) for intent in intents_ba]
        int_ext_map = dict(zip(intents_ba, extents_ba))

    def get_vals_for_column(column_name):
        def verbalise_descriptions(bas):
            return [verbalise(ba, attributes) for ba in bas]

        if column_name == 'premise':
            return verbalise_descriptions(premises)
        if column_name == 'conclusion':
            return verbalise_descriptions(conclusions)
        if column_name == 'conclusion_full':
            return verbalise_descriptions(map(intents_ba.__getitem__, intents_idxs))
        if column_name == 'extent':
            return [verbalise(int_ext_map[intents_ba[intent_i]], objects) for intent_i in intents_idxs]
        if column_name == 'support':
            return [int_ext_map[intents_ba[intent_i]].count() for intent_i in intents_idxs]
        raise NotImplementedError("Something's gone wrong in the code")

    impls_df = pd.DataFrame({f: get_vals_for_column(f) for f in to_compute})
    if unit_base:
        impls_df['conclusion'] = [list(conclusion)[0] for conclusion in impls_df['conclusion']]
    return impls_df

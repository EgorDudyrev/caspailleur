"""Module with easy to use general functions for working with Caspailleur"""
from operator import itemgetter
import typing
from typing import Iterator, Iterable, Literal, Union, Optional
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
        n_most_stable: int = None,
) -> pd.DataFrame:
    bitarrays, objects, attributes = to_bitarrays(data)

    n_objects = len(objects)
    min_support = to_absolute_number(min_support, n_objects)

    attr_extents = transpose_context(bitarrays)
    descriptions = None
    if n_most_stable is None:
        intents_ba = mec.list_intents_via_LCM(bitarrays, min_supp=min_support)
        extents_ba = [fbarray(extension(intent, attr_extents)) for intent in intents_ba]
        keys_ba = mec.list_keys(intents_ba)
        passkeys_ba = mec.list_passkeys(intents_ba)
        ppremises_ba = dict(ibases.iter_proper_premises_via_keys(intents_ba, keys_ba))
        pintents_ba = dict(ibases.list_pseudo_intents_via_keys(ppremises_ba.items(), intents_ba))

    else:  # n_most_stable is defined
        extents_ba = mec.list_stable_extents_via_sofia(
            attr_extents, n_most_stable, min_supp=min_support,
            use_tqdm=False, n_attributes=len(attributes)
        )
        intents_ba = [fbarray(intention(extent, attr_extents)) for extent in extents_ba]
        intents_ba, old_new_id_map = topological_sorting(intents_ba)
        extents_ba = list(map(itemgetter(1), sorted(zip(old_new_id_map, extents_ba), key=itemgetter(0))))
        descriptions = to_itemsets(intents_ba)[0]

    ext_int_map = dict(zip(extents_ba, intents_ba))

    descriptions = list(map(set, powerset(range(len(attributes))))) if descriptions is None else descriptions
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

    # TODO: Compute keys and passkeys for stable concepts
    if n_most_stable is None:
        descr_df['is_key'] = [descr_ba in keys_ba for descr_ba in descr_df['descr_ba']]
        descr_df['is_passkey'] = [descr_ba in passkeys_ba for descr_ba in descr_df['descr_ba']]

    if n_most_stable is None:  # otherwise, I do not know how to compute implication bases
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
        to_compute: Optional[Union[list[MINE_CONCEPTS_COLUMN], Literal['all']]] = ('extent', 'intent', 'support', 'delta-stability', 'keys', 'passkeys', 'proper_premises'),
        min_support: Union[int, float] = 0,
        use_tqdm: bool = False,
        return_all_computed: bool = False
) -> pd.DataFrame:
    def group_by_concept(pairs: Iterable[tuple[fbarray, int]], n_cncpts: int, attrs: list[str]) -> list[list[set[str]]]:
        per_concept = [[] for _ in range(n_cncpts)]
        for ba, cncpt_i in pairs:
            per_concept[cncpt_i].append(verbalise(ba, attrs))
        return per_concept

    all_cols = list(typing.get_args(MINE_CONCEPTS_COLUMN))
    cols_to_compute = set(to_compute if to_compute and to_compute != 'all' else all_cols)
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

    cols_to_return = all_cols
    if to_compute and to_compute != 'all':
        cols_to_return = sorted(cols_to_compute, key=all_cols.index) if return_all_computed else list(to_compute)

    bitarrays, objects, attributes = to_bitarrays(data)
    attr_extents = transpose_context(bitarrays)

    n_objects = len(objects)
    min_support = to_absolute_number(min_support, n_objects)

    concepts_df = pd.DataFrame()
    if 'intent' in cols_to_compute:
        intents_ba = mec.list_intents_via_LCM(bitarrays, min_supp=min_support)
        concepts_df['intent'] = [verbalise(intent_ba, attributes) for intent_ba in intents_ba]
        n_concepts = len(intents_ba)
    if 'extent' in cols_to_compute:
        extents_ba = [fbarray(extension(intent, attr_extents)) for intent in intents_ba]
        concepts_df['extent'] = [verbalise(extent_ba, objects) for extent_ba in extents_ba]
    if 'support' in cols_to_compute:
        concepts_df['support'] = [idxs.support_by_description(..., ..., extent_ba) for extent_ba in extents_ba]
    if 'delta-stability' in cols_to_compute:
        concepts_df['delta-stability'] = [idxs.delta_stability_by_description(descr, attr_extents, extent_ba)
                                          for descr, extent_ba in zip(intents_ba, extents_ba)]
    if 'keys' in cols_to_compute:
        keys_ba = mec.list_keys(intents_ba)
        concepts_df['keys'] = group_by_concept(keys_ba.items(), n_concepts, attributes)
    if 'passkeys' in cols_to_compute:
        passkeys_ba = mec.list_passkeys(intents_ba)
        concepts_df['passkeys'] = group_by_concept(passkeys_ba.items(), n_concepts, attributes)
    if 'proper_premises' in cols_to_compute:
        ppremises_ba = dict(ibases.iter_proper_premises_via_keys(intents_ba, keys_ba))
        concepts_df['proper_premises'] = group_by_concept(ppremises_ba.items(), n_concepts, attributes)
    if 'pseudo_intents' in cols_to_compute:
        pintents_ba = dict(ibases.list_pseudo_intents_via_keys(
            ppremises_ba.items(), intents_ba, use_tqdm=use_tqdm, n_keys=len(ppremises_ba)))
        concepts_df['pseudo_intents'] = group_by_concept(pintents_ba.items(), n_concepts, attributes)

    return concepts_df[cols_to_return]


BASIS_NAME = Literal[
    "Proper Premise", "Canonical Direct", "Karell",
    "Pseudo-Intent", "Canonical", "Duquenne-Guigues", "Minimum",
]


def mine_implications(
        data: ContextType, basis_name: BASIS_NAME = 'Proper Premise',
        unit_base: bool = False
) -> pd.DataFrame:
    assert basis_name in BASIS_NAME.__args__,\
        f"You asked for '{basis_name}' basis. But only the following bases are supported: {BASIS_NAME}"
    if basis_name in {'Canonical Direct', "Karell"}:
        basis_name = 'Proper Premise'
    if basis_name in {'Canonical', 'Duquenne-Guigues', 'Minimum'}:
        basis_name = 'Pseudo-Intent'

    bitarrays, objects, attributes = to_bitarrays(data)
    attr_extents = transpose_context(bitarrays)

    intents_ba = mec.list_intents_via_LCM(bitarrays)
    extents_ba = [fbarray(extension(intent, attr_extents)) for intent in intents_ba]
    int_ext_map = dict(zip(intents_ba, extents_ba))
    keys_ba = mec.list_keys(intents_ba)
    ppremises_ba = list(ibases.iter_proper_premises_via_keys(intents_ba, keys_ba))
    basis = ppremises_ba
    if basis_name == 'Pseudo-Intent':
        basis = ibases.list_pseudo_intents_via_keys(ppremises_ba, intents_ba)

    pseudo_closures = [
        ibases.saturate(premise, basis[:impl_i]+basis[impl_i+1:], intents_ba)
        for impl_i, (premise, intent_i) in enumerate(basis)
    ]
    basis = [(premise, intents_ba[intent_i] & ~pintent, intent_i)
             for (premise, intent_i), pintent in zip(basis, pseudo_closures)]

    if unit_base:
        single_attrs = to_bitarrays([{i} for i in range(len(attributes))])[0]
        basis = [(premise, single_attrs[attr_i], intent_i)
                 for premise, conclusion, intent_i in basis for attr_i in conclusion.search(True)]
    premises, conclusions, intents_idxs = zip(*basis)

    impls_df = pd.DataFrame({
        'premise': premises,
        'conclusion': conclusions,
        'conclusion_full': [intents_ba[intent_i] for intent_i in intents_idxs],
        'extent': [int_ext_map[intents_ba[intent_i]] for intent_i in intents_idxs],
        'support': [int_ext_map[intents_ba[intent_i]].count() for intent_i in intents_idxs]
    })
    for ba_col in ['premise', 'conclusion', 'conclusion_full', 'extent']:
        impls_df[ba_col] = impls_df[ba_col].map(
            lambda bitarray: verbalise(bitarray, attributes if ba_col != 'extent' else objects))

    if unit_base:
        impls_df['conclusion'] = [list(conclusion)[0] for conclusion in impls_df['conclusion']]
    return impls_df

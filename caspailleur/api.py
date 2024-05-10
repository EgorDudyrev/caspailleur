"""Module with easy to use general functions for working with Caspailleur"""
from typing import Iterator, Iterable, Literal
import pandas as pd
from bitarray import frozenbitarray as fbarray

import io
from .base_functions import powerset, extension
from .io import ContextType, to_bitarrays, transpose_context, isets2bas, verbalise
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


def mine_descriptions(data: ContextType) -> pd.DataFrame:
    bitarrays, objects, attributes = to_bitarrays(data)
    attr_extents = transpose_context(bitarrays)

    intents_ba = mec.list_intents_via_LCM(bitarrays)
    extents_ba = [fbarray(extension(intent, attr_extents)) for intent in intents_ba]
    ext_int_map = dict(zip(extents_ba, intents_ba))
    keys_ba = mec.list_keys(intents_ba)
    passkeys_ba = mec.list_passkeys(intents_ba)
    ppremises_ba = dict(ibases.iter_proper_premises_via_keys(intents_ba, keys_ba))
    pintents_ba = dict(ibases.list_pseudo_intents_via_keys(ppremises_ba.items(), intents_ba))

    descr_df = pd.DataFrame(index=pd.Series(list(map(set, powerset(range(len(attributes))))), name='description_idxs'))
    descr_df['description'] = [set(verbalise(descr_idxs, attributes)) for descr_idxs in descr_df.index]
    descr_df['extent_ba'] = [extension(descr, attr_extents) for descr in descr_df.index]
    descr_df['extent'] = [set(verbalise(extent_ba, objects)) for extent_ba in descr_df['extent_ba']]
    descr_df['intent'] = [set(verbalise(ext_int_map[extent_ba], attributes)) for extent_ba in descr_df['extent_ba']]
    descr_df['support'] = [idxs.support_by_description(..., ..., extent) for extent in descr_df['extent_ba']]
    descr_df['delta-stability'] = [idxs.delta_stability_by_description(descr, attr_extents, extent)
                                   for descr, extent in descr_df['extent_ba'].items()]
    descr_df['is_closed'] = descr_df['intent'] == descr_df['description']
    descr_df['descr_ba'] = list(isets2bas(descr_df.index, len(attributes)))
    descr_df['is_key'] = [descr_ba in keys_ba for descr_ba in descr_df['descr_ba']]
    descr_df['is_passkey'] = [descr_ba in passkeys_ba for descr_ba in descr_df['descr_ba']]
    descr_df['is_proper_premise'] = [descr_ba in ppremises_ba for descr_ba in descr_df['descr_ba']]
    descr_df['is_pseudo_intent'] = [descr_ba in pintents_ba for descr_ba in descr_df['descr_ba']]

    descr_df = descr_df.drop(columns=['extent_ba', 'descr_ba']).reset_index(drop=True)
    return descr_df


def mine_concepts(data: ContextType) -> pd.DataFrame:
    def group_by_concept(pairs: Iterable[tuple[fbarray, int]], n_cncpts: int, attrs: list[str]) -> list[list[set[str]]]:
        per_concept = [[] for _ in range(n_cncpts)]
        for ba, cncpt_i in pairs:
            per_concept[cncpt_i].append(verbalise(ba, attrs))
        return per_concept

    bitarrays, objects, attributes = to_bitarrays(data)
    attr_extents = transpose_context(bitarrays)

    intents_ba = mec.list_intents_via_LCM(bitarrays)
    extents_ba = [fbarray(extension(intent, attr_extents)) for intent in intents_ba]
    keys_ba = mec.list_keys(intents_ba)
    passkeys_ba = mec.list_passkeys(intents_ba)
    ppremises_ba = dict(ibases.iter_proper_premises_via_keys(intents_ba, keys_ba))
    pintents_ba = dict(ibases.list_pseudo_intents_via_keys(ppremises_ba.items(), intents_ba))

    n_concepts = len(intents_ba)

    concepts_df = pd.DataFrame()
    concepts_df['extent'] = [verbalise(extent_ba, objects) for extent_ba in extents_ba]
    concepts_df['intent'] = [verbalise(intent_ba, attributes) for intent_ba in intents_ba]
    concepts_df['support'] = [idxs.support_by_description(..., ..., extent_ba) for extent_ba in extents_ba]
    concepts_df['delta-stability'] = [idxs.delta_stability_by_description(descr, attr_extents, extent_ba)
                                      for descr, extent_ba in zip(intents_ba, extents_ba)]

    concepts_df['keys'] = group_by_concept(keys_ba.items(), n_concepts, attributes)
    concepts_df['passkeys'] = group_by_concept(passkeys_ba.items(), n_concepts, attributes)
    concepts_df['proper_premises'] = group_by_concept(ppremises_ba.items(), n_concepts, attributes)
    concepts_df['pseudo_intents'] = group_by_concept(pintents_ba.items(), n_concepts, attributes)
    return concepts_df


def mine_implications(data: ContextType, basis_name: Literal['proper premise', 'pseudo-intent'] = 'proper premise') -> pd.DataFrame:
    bitarrays, objects, attributes = to_bitarrays(data)
    attr_extents = transpose_context(bitarrays)

    intents_ba = mec.list_intents_via_LCM(bitarrays)
    extents_ba = [fbarray(extension(intent, attr_extents)) for intent in intents_ba]
    int_ext_map = dict(zip(intents_ba, extents_ba))
    keys_ba = mec.list_keys(intents_ba)
    ppremises_ba = list(ibases.iter_proper_premises_via_keys(intents_ba, keys_ba))
    if basis_name == 'pseudo-intent':
        basis = ibases.list_pseudo_intents_via_keys(ppremises_ba, intents_ba)
    elif basis_name == 'proper premise':
        basis = ppremises_ba
    else:
        raise ValueError('Only "proper premise" and "pseudo-intent" bases are supported')

    impls_df = pd.DataFrame()
    impls_df['premise'] = [verbalise(premise, attributes) for premise, _ in basis]
    impls_df['conclusion'] = [verbalise(intents_ba[intent_i] & (~premise), attributes) for premise, intent_i in basis]
    impls_df['conclusion_full'] = [verbalise(intents_ba[intent_i], attributes) for _, intent_i in basis]
    impls_df['extent'] = [verbalise(int_ext_map[intents_ba[intent_i]], objects) for _, intent_i in basis]
    impls_df['support'] = [idxs.support_by_description(..., ..., int_ext_map[intents_ba[intent_i]])
                           for _, intent_i in basis]
    return impls_df

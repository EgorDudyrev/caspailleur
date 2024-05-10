"""Module with easy to use general functions for working with Caspailleur"""
from typing import Iterator
import pandas as pd
from bitarray import frozenbitarray as fbarray

from .base_functions import powerset, extension
from .io import ContextType, to_bitarrays, transpose_context
from . import indices as idxs
from . import definitions


def iter_descriptions(data: ContextType) -> Iterator[dict]:
    bitarrays, objects, attributes = to_bitarrays(data)
    attr_extents = transpose_context(bitarrays)

    sub_pseudo_intents = []

    for description_idxs in powerset(range(len(attributes))):
        extent_ba = fbarray(extension(description_idxs, attr_extents))
        stats = {
            'description': {attributes[attr_i] for attr_i in description_idxs},
            'extent': {objects[obj_i] for obj_i in extent_ba.itersearch(True)},
            'closure': {attributes[attr_i] for attr_i in definitions.closure(description_idxs, attr_extents)},
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
    return pd.DataFrame(iter_descriptions(data))


def mine_concepts(data: ContextType) -> pd.DataFrame:
    pass


def mine_implications(data: ContextType, basis_name: str = 'proper premise') -> pd.DataFrame:
    pass
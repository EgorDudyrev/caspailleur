"""Module with easy to use general functions for working with Caspailleur"""
from typing import Iterator
import pandas as pd
from bitarray import frozenbitarray as fbarray

from .base_functions import powerset, extension
from .io import ContextType, to_bitarrays, transpose_context
from . import indices as idxs


def iter_descriptions(data: ContextType) -> Iterator[dict]:
    bitarrays, objects, attributes = to_bitarrays(data)
    attr_extents = transpose_context(bitarrays)
    
    for description_idxs in powerset(range(len(attributes))):
        extent_ba = fbarray(extension(description_idxs, attr_extents))
        yield {
            'description': {attributes[attr_i] for attr_i in description_idxs}, 
            'extent': {objects[obj_i] for obj_i in extent_ba.itersearch(True)},
            'support': idxs.support_by_description(description_idxs, attr_extents, extent_ba),
            'delta-stability': idxs.delta_stability_by_description(description_idxs, attr_extents)
        }


def mine_descriptions(data: ContextType) -> pd.DataFrame:
    return pd.DataFrame(iter_descriptions(data))


def mine_concepts(data: ContextType) -> pd.DataFrame:
    pass


def mine_implications(data: ContextType, basis_name: str = 'proper premise') -> pd.DataFrame:
    pass
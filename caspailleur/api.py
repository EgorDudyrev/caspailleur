"""Module with easy to use general functions for working with Caspailleur"""
from typing import Iterator
import pandas as pd

from .base_functions import powerset, extension
from .io import ContextType, to_itemsets, transpose_context


def iter_descriptions(data: ContextType) -> Iterator[dict]:
    itemsets, objects, attributes = to_itemsets(data)
    attr_extents = transpose_context(itemsets)
    
    for description_idxs in powerset(range(len(attributes))):
        extent_idxs = extension(description_idxs, attr_extents)
        yield {
            'description': {attributes[attr_i] for attr_i in description_idxs}, 
            'extent': {objects[obj_i] for obj_i in extent_idxs},
            'support': len(extent_idxs)
        }


def mine_descriptions(data: ContextType) -> pd.DataFrame:
    return pd.DataFrame(iter_descriptions(data)).set_index('description')


def mine_concepts(data: ContextType) -> pd.DataFrame:
    pass


def mine_implications(data: ContextType, basis_name: str = 'proper premise') -> pd.DataFrame:
    pass
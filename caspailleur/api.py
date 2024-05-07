"""Module with easy to use general functions for working with Caspailleur"""
from typing import Iterator
import pandas as pd

from .base_functions import powerset
from .io import ContextType, to_itemsets, to_pandas


def iter_descriptions(data: ContextType) -> Iterator[dict]:
    df = to_pandas(data)
    for description in powerset(df.columns):
        extent = set(df.index[df[list(description)].all(1)].to_list()) if description else set(df.index.to_list())
        yield {
            'description': set(description), 
            'extent': extent,
            'support': len(extent)
        }


def mine_descriptions(data: ContextType) -> pd.DataFrame:
    return pd.DataFrame(iter_descriptions(data)).set_index('description')


def mine_concepts(data: ContextType) -> pd.DataFrame:
    pass


def mine_implications(data: ContextType, basis_name: str = 'proper premise') -> pd.DataFrame:
    pass
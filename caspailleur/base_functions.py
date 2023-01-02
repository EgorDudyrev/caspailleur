from itertools import chain, combinations
from typing import Iterable, FrozenSet, List, Dict, Iterator
import numpy.typing as npt

import numpy as np
from bitarray import frozenbitarray as fbarray
from bitarray.util import zeros as bazeros


################
# Math notions #
################
def powerset(iterable) -> Iterable[FrozenSet[int]]:
    """powerset({1,2,3}) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return map(frozenset, chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))


def is_subset_of(A: FrozenSet[int], B: FrozenSet[int]) -> bool:
    """Test whether `A` is a subset of `B`"""
    return A & B == A


def is_psubset_of(A: FrozenSet[int], B: FrozenSet[int]) -> bool:
    """Test whether `A` is a proper subset of `B`"""
    return (A & B == A) and A != B


def closure(B: Iterator[int], crosses_per_columns: List[FrozenSet[int]]) -> Iterator[int]:
    n_rows = max(max(col) + 1 for col in crosses_per_columns if col)

    extent = set(range(n_rows))
    for m in B:
        extent &= crosses_per_columns[m]

    intent = (m for m, col in enumerate(crosses_per_columns) if is_subset_of(extent, col))
    return intent


##########################
# Basic type conversions #
##########################
def np2isets(X: npt.NDArray[np.int_]) -> List[npt.NDArray[np.int_]]:
    return [np.array(row.nonzero()[0]) for row in X]


def iset2ba(itemset: Iterator[int], length: int) -> fbarray:
    bar = bazeros(length)
    for m in itemset:
        bar[m] = True
    return fbarray(bar)


def ba2iset(bar: fbarray) -> Iterator[int]:
    return bar.itersearch(1)

from itertools import chain, combinations
from typing import Iterable, Iterator, List, FrozenSet, Union, Any


from bitarray import frozenbitarray as fbarray


################
# Math notions #
################
def powerset(iterable: Iterable[Any]) -> Iterable[FrozenSet[Any]]:
    """Iterate through all subsets of elements of set `iterable`

    Examples
    --------
    powerset({1,2,3}) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    s = list(iterable)
    return map(frozenset, chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))


def is_subset_of(A: Union[FrozenSet[int], fbarray], B: Union[FrozenSet[int], fbarray]) -> bool:
    """Test whether `A` is a subset of `B`"""
    return A & B == A


def is_psubset_of(A: Union[FrozenSet[int], fbarray], B: Union[FrozenSet[int], fbarray]) -> bool:
    """Test whether `A` is a proper subset of `B`"""
    return (A & B == A) and A != B


def extension(description: Iterable[int], crosses_per_columns: List[FrozenSet[int]]) -> FrozenSet[int]:
    """Select the indices of rows described by `description`"""
    n_rows = max(max(col) + 1 for col in crosses_per_columns if col)
    extent = set(range(n_rows))
    for m in description:
        extent &= crosses_per_columns[m]
    return frozenset(extent)


def intention(objects: Iterable[int], crosses_per_columns: List[FrozenSet[int]]) -> Iterator[int]:
    """Iterate the indices of columns that describe the `objects`"""
    return (m for m, col in enumerate(crosses_per_columns) if is_subset_of(objects, col))


def closure(description: Iterable[int], crosses_per_columns: List[FrozenSet[int]]) -> Iterator[int]:
    """Iterate indices of all columns who describe the same rows as `description`

    Parameters
    ----------
    description: Iterable[int]
        Indices of some columns from `crosses_per_columns`
    crosses_per_columns: List[FrozenSet[int]]
        List of indices of 'True' rows for each column

    Returns
    -------
    intent: Iterator[int]
        Indices of All columns with the same intersection as `D`

    """
    return intention(extension(description, crosses_per_columns), crosses_per_columns)

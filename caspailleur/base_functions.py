from functools import reduce
from itertools import chain, combinations
from typing import Iterable, Iterator, Union, Any

from bitarray import frozenbitarray as fbarray, bitarray

from . import io


################
# Math notions #
################
def powerset(iterable: Iterable[Any]) -> Iterable[frozenset[Any]]:
    """Iterate through all subsets of elements of set `iterable`

    Examples
    --------
    powerset({1,2,3}) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    s = list(iterable)
    return map(frozenset, chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))


def is_subset_of(A: Union[set[int], fbarray], B: Union[set[int], fbarray]) -> bool:
    """Test whether `A` is a subset of `B`"""
    return A & B == A


def is_psubset_of(A: Union[set[int], fbarray], B: Union[set[int], fbarray]) -> bool:
    """Test whether `A` is a proper subset of `B`"""
    return (A & B == A) and A != B


def maximal_extent(crosses_per_columns: Union[list[set], list[bitarray]]) -> Union[set, bitarray]:
    """Return the whole set of objects from `crosses_per_columns` data representation"""
    first_column = crosses_per_columns[0]
    if isinstance(first_column, bitarray):
        return first_column | (~first_column)
    if any(column and isinstance(next(iter(column)), int) for column in crosses_per_columns):
        n_rows = max(max(column) for column in crosses_per_columns if column) + 1
        return type(first_column)(range(n_rows))

    all_attrs = reduce(first_column.__or__ , crosses_per_columns)
    return type(first_column)(all_attrs)


def extension(description: Union[Iterable[int], bitarray], crosses_per_columns: Union[list[set], list[bitarray]])\
        -> Union[set[int], bitarray]:
    """Select the indices of rows described by `description`"""
    column_type = type(crosses_per_columns[0])
    description = description.itersearch(True) if isinstance(description, bitarray) else description

    total_extent = column_type(maximal_extent(crosses_per_columns))
    extent = reduce(column_type.__and__, (crosses_per_columns[attr_i] for attr_i in description), total_extent)
    return extent


def intention(
        objects: Union[set[int], bitarray], crosses_per_columns: Union[list[set[int]], list[bitarray]]
) -> Union[Iterator[int], bitarray]:
    """Iterate the indices of columns that describe the `objects`"""
    if isinstance(objects, bitarray):
        return type(objects)([is_subset_of(objects, col) for col in crosses_per_columns])

    if isinstance(crosses_per_columns[0], bitarray) and not isinstance(objects, bitarray):
        objects = next(io.isets2bas([objects], len(crosses_per_columns[0])))

    return (m for m, col in enumerate(crosses_per_columns) if is_subset_of(objects, col))


def closure(
        description: Union[Iterable[int], bitarray], crosses_per_columns: Union[list[set[int]], list[bitarray]]
) -> Union[Iterator[int], bitarray]:
    """Iterate indices of all columns who describe the same rows as `description`

    Parameters
    ----------
    description:
        Indices of some columns from `crosses_per_columns`
    crosses_per_columns:
        List of indices of 'True' rows for each column

    Returns
    -------
    intent:
        Indices of All columns with the same intersection as `D`

    """
    result = intention(extension(description, crosses_per_columns), crosses_per_columns)
    if not isinstance(description, bitarray) and isinstance(result, bitarray):
        return (i for i in result.itersearch(True))
    return result


##############################
# Reverse compatibility part #
##############################
from .io import isets2bas

from itertools import chain, combinations
from typing import Iterable, Iterator, List, FrozenSet, BinaryIO
import numpy.typing as npt

import numpy as np
from bitarray import frozenbitarray as fbarray, bitarray
from bitarray.util import zeros as bazeros


################
# Math notions #
################
def powerset(iterable: Iterable[int]) -> Iterable[FrozenSet[int]]:
    """Iterate through all subsets of elements of set `iterable`

    Examples
    --------
    powerset({1,2,3}) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    s = list(iterable)
    return map(frozenset, chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))


def is_subset_of(A: FrozenSet[int] or fbarray, B: FrozenSet[int] or fbarray) -> bool:
    """Test whether `A` is a subset of `B`"""
    return A & B == A


def is_psubset_of(A: FrozenSet[int] or fbarray, B: FrozenSet[int] or fbarray) -> bool:
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


##########################
# Basic type conversions #
##########################
def np2bas(X: npt.ArrayLike) -> List[fbarray]:
    """Convert numpy boolean matrix to the list of bitarrays (one per row of the matrix)"""
    return [fbarray(x) for x in X.tolist()]


def bas2np(barrays: Iterable[fbarray]) -> npt.ArrayLike:
    """Convert the list of bitarrays to numpy boolean matrix (bitarrays become rows in of the matrix)"""
    return np.vstack([ba.tolist() for ba in barrays]).astype(np.bool_)


def isets2bas(itemsets: Iterable[Iterable[int]], length: int) -> Iterator[fbarray]:
    """Convert the list of lists of indices of 'True' elements to bitarrays of given length

    Examples
    --------
    isets2bas([ [0, 1], [1,3,4] ], 5) --> [bitarray('01000'), bitarray('01011')]
    """
    for iset in itemsets:
        bar = bazeros(length)
        for m in iset:
            bar[m] = True
        yield fbarray(bar)


def bas2isets(bitarrays: Iterable[fbarray]) -> Iterator[FrozenSet[int]]:
    """Convert the list of bitarrays to the list of lists of indices of 'True' elements of the bitarrays

    Examples
    --------
    bas2isets([ bitarray('01000'), bitarray('01011') ]) --> [ [0,1], [1,3,4] ]
    """
    for bar in bitarrays:
        yield frozenset(bar.itersearch(True))


###########################
# Save and load functions #
###########################

def load_balist(file: BinaryIO) -> Iterator[bitarray]:
    """Load the list of bitarrays from binary file"""
    basize = b''
    while True:
        data = file.read(1)
        if data == b'n':
            break
        basize += data
    basize = int(basize.decode(encoding='utf-8'))
    basize_bytes = len(bazeros(basize).tobytes())

    while True:
        data = file.read(basize_bytes)
        if data == b'':
            break

        ba = bitarray()
        ba.frombytes(data)
        yield ba[:basize]


def save_balist(file: BinaryIO, bitarrays: List[bitarray]):
    """Save the list of bitarrays to the binary file (proposed file extension '.bal')"""
    basize = len(bitarrays[0])
    assert all(len(ba) == basize for ba in bitarrays), "All bitarrays should be of the same size"

    file.write(str(basize).encode(encoding='utf-8'))
    file.write(b'n')

    for ba in bitarrays:
        file.write(ba.tobytes())

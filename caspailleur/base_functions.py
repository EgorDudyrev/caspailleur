from itertools import chain, combinations
from typing import Iterable, Iterator, List, FrozenSet, BinaryIO
import numpy.typing as npt

import numpy as np
from bitarray import frozenbitarray as fbarray, bitarray
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
def np2bas(X: npt.ArrayLike) -> List[fbarray]:
    return [fbarray(x) for x in X.tolist()]


def bas2np(barrays: Iterable[fbarray]) -> npt.ArrayLike:
    return np.vstack([ba.tolist() for ba in barrays]).astype(np.bool_)


def isets2bas(itemsets: Iterable[Iterable[int]], length: int) -> Iterator[fbarray]:
    for iset in itemsets:
        bar = bazeros(length)
        for m in iset:
            bar[m] = True
        yield fbarray(bar)


def bas2isets(bitarrays: Iterable[fbarray]) -> Iterator[FrozenSet[int]]:
    for bar in bitarrays:
        yield frozenset(bar.itersearch(True))


###########################
# Save and load functions #
###########################

def load_balist(file: BinaryIO) -> Iterator[bitarray]:
    basize = b''
    while True:
        data = file.read(1)
        if data == b'\n':
            break
        basize += data
    basize = int(basize.decode())

    while True:
        data = file.read(basize)
        if data == b'':
            break

        ba = bitarray()
        ba.frombytes(data)
        yield ba[:basize]


def save_balist(file: BinaryIO, bitarrays: List[bitarray]):
    basize = len(bitarrays[0])
    assert all(len(ba) == basize for ba in bitarrays), "All bitarrays should be of the same size"

    file.write(str(basize).encode())
    file.write(b'\n')

    for ba in bitarrays:
        file.write(ba.tobytes())

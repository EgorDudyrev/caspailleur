from itertools import chain, combinations
from typing import Iterable, FrozenSet, List


def powerset(iterable) -> Iterable[FrozenSet[int]]:
    """powerset({1,2,3}) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return map(frozenset, chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))


def is_subset_of(A: FrozenSet[int], B: FrozenSet[int]):
    """Test whether `A` is a subset of `B`"""
    return A & B == A


def is_psubset_of(A: FrozenSet[int], B: FrozenSet[int]):
    """Test whether `A` is a proper subset of `B`"""
    return (A & B == A) and A != B


def closure(B: FrozenSet[int], crosses_per_columns: List[FrozenSet[int]]) -> FrozenSet[int]:
    n_rows = max(max(col) + 1 for col in crosses_per_columns if col)

    extent = set(range(n_rows))
    for m in B:
        extent &= crosses_per_columns[m]

    intent = frozenset({m for m, col in enumerate(crosses_per_columns) if m in B or is_subset_of(extent, col)})
    return intent

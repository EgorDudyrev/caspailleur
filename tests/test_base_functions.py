from bitarray import frozenbitarray as fbarray
import numpy as np

from caspailleur import base_functions as bfunc


def test_powerset():
    iterable = (1, 2, 3)
    pset_true = [[], [1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]]
    pset_true = [frozenset(ar) for ar in pset_true]

    pset = bfunc.powerset(iterable)
    assert list(pset) == pset_true


def test_is_subset_of():
    assert bfunc.is_subset_of(set(), {1, 2, 3})
    assert bfunc.is_subset_of({2, 5, 3}, {2, 1, 3, 5})
    assert not bfunc.is_subset_of({2, 1, 3, 5}, {1})
    assert bfunc.is_subset_of({1, 2, 3}, {1, 2, 3})
    assert not bfunc.is_subset_of({0, 1, 3}, {0, 1, 2})


def test_is_psubset_of():
    assert bfunc.is_psubset_of(set(), {1, 2, 3})
    assert bfunc.is_psubset_of({2, 5, 3}, {2, 1, 3, 5})
    assert not bfunc.is_psubset_of({2, 1, 3, 5}, {1})
    assert not bfunc.is_psubset_of({1, 2, 3}, {1, 2, 3})


def test_closure():
    crosses_per_columns = [
        {0, 1,    3},  # col 0
        {0, 1, 2   },  # col 1
        {0,    2   },  # col 2
        {0, 1, 2, 3},  # col 3
        set(),         # col 4
    ]

    assert list(bfunc.closure([0], crosses_per_columns)) == [0, 3]
    assert list(bfunc.closure([0, 1], crosses_per_columns)) == [0, 1, 3]
    assert list(bfunc.closure([1, 2], crosses_per_columns)) == [1, 2, 3]
    assert list(bfunc.closure([], crosses_per_columns)) == [3]


def test_np_ba_conversions():
    X = np.array([[True, True, False], [False, False, True]])
    bars_true = [fbarray(x) for x in X.tolist()]
    bars = bfunc.np2bas(X)
    assert bars == bars_true

    X_reconstr = bfunc.bas2np(bars)
    assert (X == X_reconstr).all()

    attr_exts_true = [fbarray([True, False]), fbarray([True, False]), fbarray([False, True])]
    assert bfunc.np2bas(X.T) == attr_exts_true


def test_iset_ba_conversions():
    iset, l = {0, 1}, 5
    ba = fbarray([True, True, False, False, False])

    assert list(bfunc.isets2bas([iset], l)) == [ba]
    assert list(bfunc.bas2isets([ba])) == [iset]
    assert list(bfunc.bas2isets(bfunc.isets2bas([iset], l))) == [iset]


def test_conversion_pipeline():
    X = np.array([[True, True, False], [False, False, True]])
    itemsets_true = [{0, 1}, {2}]
    itemsets = list(bfunc.bas2isets(bfunc.np2bas(X)))
    assert itemsets == itemsets_true

    X_reconstr = bfunc.bas2np(bfunc.isets2bas(itemsets, 3))
    assert (X == X_reconstr).all()

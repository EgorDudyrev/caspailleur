import bitarray
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


def test_np2isets():
    X = np.array([[True, True, False], [False, False, True]])
    isets_true = [np.array([0, 1]), np.array([2])]
    isets = bfunc.np2isets(X)
    assert all([(iset == iset_true).all() for iset, iset_true in zip(isets, isets_true)])


def test_iset_ba_conversions():
    iset, l = [0, 1], 5
    ba = bitarray.frozenbitarray([True, True, False, False, False])

    assert bfunc.iset2ba(iset, l) == ba
    assert list(bfunc.ba2iset(ba)) == iset
    assert bfunc.iset2ba(bfunc.ba2iset(ba), len(ba)) == ba


def test_iter_attribute_extents():
    K = np.array([
        [True, False, False, True, False],
        [True, False, True, False, False],
        [False, True, True, False, False],
        [False, True, True, True, False],
    ])
    attr_extents_true = [bfunc.iset2ba(iset, 4) for iset in [
        [0, 1], [2, 3], [1, 2, 3], [0, 3], []
    ]]
    attr_extents = list(bfunc.iter_attribute_extents(K))

    assert attr_extents == attr_extents_true


def test_conversion_pipeline():
    X = np.array([[True, True, False], [False, False, True]])
    itemsets_true = [[0, 1], [2]]
    itemsets = bfunc.np2isets(X)
    assert itemsets == itemsets_true

    barrays_true = [bitarray.frozenbitarray([bool(v) for v in x]) for x in X]
    barrays = [bfunc.iset2ba(iset, X.shape[1]) for iset in itemsets]
    assert barrays == barrays_true

    itemsets = [list(bfunc.ba2iset(barray)) for barray in barrays]
    assert itemsets == itemsets_true

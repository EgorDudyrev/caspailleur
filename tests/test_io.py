import numpy as np
from bitarray import frozenbitarray as fbarray
import os

from caspailleur import io



def test_np_ba_conversions():
    X = np.array([[True, True, False], [False, False, True]])
    bars_true = [fbarray(x) for x in X.tolist()]
    bars = io.np2bas(X)
    assert bars == bars_true

    X_reconstr = io.bas2np(bars)
    assert (X == X_reconstr).all()

    attr_exts_true = [fbarray([True, False]), fbarray([True, False]), fbarray([False, True])]
    assert io.np2bas(X.T) == attr_exts_true


def test_iset_ba_conversions():
    iset, l = {0, 1}, 5
    ba = fbarray([True, True, False, False, False])

    assert list(io.isets2bas([iset], l)) == [ba]
    assert list(io.bas2isets([ba])) == [iset]
    assert list(io.bas2isets(io.isets2bas([iset], l))) == [iset]


def test_conversion_pipeline():
    X = np.array([[True, True, False], [False, False, True]])
    itemsets_true = [{0, 1}, {2}]
    itemsets = list(io.bas2isets(io.np2bas(X)))
    assert itemsets == itemsets_true

    X_reconstr = io.bas2np(io.isets2bas(itemsets, 3))
    assert (X == X_reconstr).all()


def test_save_load_barrays():
    bitarrays = [fbarray([True, False]), fbarray([True, False]), fbarray([False, True])]
    with open('tst.bal', 'wb') as file:
        io.save_balist(file, bitarrays)
    with open('tst.bal', 'rb') as file:
        bitarrays_load = list(io.load_balist(file))

    os.remove('tst.bal')

    assert bitarrays_load == bitarrays

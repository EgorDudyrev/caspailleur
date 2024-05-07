import numpy as np
import pandas as pd
from bitarray import frozenbitarray as fbarray
import os
import pytest

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


def test_to_itemsets():
    itemsets_true = [frozenset({0, 1}), frozenset({2})]
    objects_true = ['object_0', 'object_1']
    attributes_true = ['attribute_0', 'attribute_1', 'attribute_2']

    df = pd.DataFrame([[True, True, False], [False, False, True]],
                      index=['object_0', 'object_1'], columns=['attribute_0', 'attribute_1', 'attribute_2'])
    itemsets, objects, attributes = io.to_itemsets(df)
    assert itemsets == itemsets_true
    assert objects == objects_true
    assert attributes == attributes_true

    dct = {'object_0': {'attribute_0', 'attribute_1'}, 'object_1': {'attribute_2'}}
    itemsets, objects, attributes = io.to_itemsets(dct)
    assert itemsets == itemsets_true
    assert objects == objects_true
    assert attributes == attributes_true

    bools = [[True, True, False], [False, False, True]]
    itemsets, objects, attributes = io.to_itemsets(bools)
    assert itemsets == itemsets_true
    assert objects == objects_true
    assert attributes == attributes_true

    itsets = [[0, 1], [2]]
    itemsets, objects, attributes = io.to_itemsets(itsets)
    assert itemsets == itemsets_true
    assert objects == objects_true
    assert attributes == attributes_true

    with pytest.raises(io.UnknownContextTypeError):
        io.to_itemsets('hello world')

    assert io.to_itemsets([]) == ([], [], [])


def test_to_dictionary():
    dct_true = {'object_0': frozenset({'attribute_0', 'attribute_1'}), 'object_1': frozenset({'attribute_2'})}
    
    df = pd.DataFrame([[True, True, False], [False, False, True]],
                      index=['object_0', 'object_1'], columns=['attribute_0', 'attribute_1', 'attribute_2'])
    dct = io.to_dictionary(df)
    assert dct == dct_true
    
    dct_ = {'object_0': {'attribute_0', 'attribute_1'}, 'object_1': {'attribute_2'}}
    dct = io.to_dictionary(dct_)
    assert dct == dct_true

    bools = [[True, True, False], [False, False, True]]
    dct = io.to_dictionary(bools)
    assert dct == dct_true

    itsets = [[0, 1], [2]]
    dct = io.to_dictionary(itsets)
    assert dct == dct_true

    with pytest.raises(io.UnknownContextTypeError):
        io.to_dictionary('hello world')

    assert io.to_dictionary([]) == dict()


def test_to_bitarrays():
    bas_true = [fbarray('110'), fbarray('001')]
    objects_true = ['object_0', 'object_1']
    attributes_true = ['attribute_0', 'attribute_1', 'attribute_2']

    
    df = pd.DataFrame([[True, True, False], [False, False, True]],
                      index=['object_0', 'object_1'], columns=['attribute_0', 'attribute_1', 'attribute_2'])
    bas, objects, attributes = io.to_bitarrays(df)
    assert bas == bas_true
    assert objects == objects_true
    assert attributes == attributes_true
    
    dct = {'object_0': {'attribute_0', 'attribute_1'}, 'object_1': {'attribute_2'}}
    bas, objects, attributes = io.to_bitarrays(dct)
    assert bas == bas_true
    assert objects == objects_true
    assert attributes == attributes_true

    bools = [[True, True, False], [False, False, True]]
    bas, objects, attributes = io.to_bitarrays(bools)
    assert bas == bas_true
    assert objects == objects_true
    assert attributes == attributes_true

    itsets = [[0, 1], [2]]
    bas, objects, attributes = io.to_bitarrays(itsets)
    assert bas == bas_true
    assert objects == objects_true
    assert attributes == attributes_true
    
    with pytest.raises(io.UnknownContextTypeError):
        io.to_bitarrays('hello world')

    assert io.to_bitarrays([]) == ([], [], [])


def test_to_pandas():
    df_true = pd.DataFrame([[True, True, False], [False, False, True]],
                      index=['object_0', 'object_1'], columns=['attribute_0', 'attribute_1', 'attribute_2'])

    df_ = pd.DataFrame([[True, True, False], [False, False, True]],
                      index=['object_0', 'object_1'], columns=['attribute_0', 'attribute_1', 'attribute_2'])
    df = io.to_pandas(df_)
    assert (df == df_true).all(None)
    
    dct = {'object_0': {'attribute_0', 'attribute_1'}, 'object_1': {'attribute_2'}}
    df = io.to_pandas(dct)
    assert (df == df_true).all(None)

    bools = [[True, True, False], [False, False, True]]
    df = io.to_pandas(bools)
    assert (df == df_true).all(None)

    itsets = [[0, 1], [2]]
    df = io.to_pandas(itsets)
    assert (df == df_true).all(None)

    with pytest.raises(io.UnknownContextTypeError):
        io.to_pandas('hello world')


    assert (io.to_pandas([]) == pd.DataFrame()).all(None)

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


def test_context_conversions():
    datas = dict(
        pandas=pd.DataFrame([[True, True, False], [False, False, True]],
                            index=['g1', 'g2'], columns=['m1', 'm2', 'm3']),
        itemset=[{0, 1}, {2}],
        named_itemset=([{0, 1}, {2}], ['g1', 'g2'], ['m1', 'm2', 'm3']),
        bitarray=[fbarray('110'), fbarray('001')],
        named_bitarray=([fbarray('110'), fbarray('001')], ['g1', 'g2'], ['m1', 'm2', 'm3']),
        bool=[[True, True, False], [False, False, True]],
        named_bool=([[True, True, False], [False, False, True]], ['g1', 'g2'], ['m1', 'm2', 'm3']),
        dict={'g1': {'m1', 'm2'}, 'g2': {'m3'}}
    )
    objects_map = {'g1': 'object_0', 'g2': 'object_1'}
    attributes_map = {'m1': 'attribute_0', 'm2': 'attribute_1', 'm3': 'attribute_2'}

    funcs = dict(
        pandas=io.to_pandas,
        itemset=io.to_itemsets,
        named_itemset=io.to_named_itemsets,
        bitarray=io.to_bitarrays,
        named_bitarray=io.to_named_bitarrays,
        bool=io.to_bools,
        named_bool=io.to_named_bools,
        dict=io.to_dictionary
    )
    named_types = {'pandas', 'named_itemset', 'named_bitarray', 'named_bool', 'dict'}

    assert datas.keys() == funcs.keys()
    for input_type in datas.keys():
        for output_type in [
            'itemset',
            'named_itemset',
            'bitarray',
            'named_bitarray',
            'bool',
            'named_bool',
            'dict'
        ]:  # datas.keys():
            input_data, output_data = datas[input_type], datas[output_type]

            # if the transformation will forget the objects/attributes names
            if (input_type not in named_types) and (output_type in named_types):
                if output_type == 'pandas':
                    output_data = output_data.rename(index=objects_map, columns=attributes_map)
                elif output_type == 'dict':
                    output_data = {objects_map[obj]: {attributes_map[attr] for attr in description}
                                  for obj, description in output_data.items()}
                else:
                    output_data = output_data[0],\
                        [objects_map[g] for g in output_data[1]], [attributes_map[m] for m in output_data[2]]

            print(input_type, output_type)
            test_data = funcs[output_type](input_data)
            assert test_data == output_data, f"Having problems transforming {input_type} context into {output_type}"


def test_transpose_context():
    df = pd.DataFrame(
        [[True, True, False], [False, False, True]],
        index=['object_0', 'object_1'], columns=['attribute_0', 'attribute_1', 'attribute_2'])
    df_transposed_true = pd.DataFrame(
        [[True, False], [True, False], [False, True]],
        index=['attribute_0', 'attribute_1', 'attribute_2'], columns=['object_0', 'object_1']
    )
    df_transposed = io.transpose_context(df)
    assert (df_transposed == df_transposed_true).all(None)
    assert (io.transpose_context(io.transpose_context(df)) == df).all(None)

    dct = {'object_0': {'attribute_0', 'attribute_1'}, 'object_1': {'attribute_2'}}    
    dct_transposed_true = {'attribute_0': {'object_0'}, 'attribute_1': {'object_0'}, 'attribute_2': {'object_1'}}
    dct_transposed = io.transpose_context(dct)
    assert dct_transposed == dct_transposed_true
    assert io.transpose_context(io.transpose_context(dct)) == dct

    bools = [[True, True, False], [False, False, True]]
    bools_transposed_true = [[True, False], [True, False], [False, True]]
    bools_transposed = io.transpose_context(bools)
    assert bools_transposed == bools_transposed_true
    assert io.transpose_context(io.transpose_context(bools)) == bools

    itsets = [{0, 1}, {2}]
    itsets_transposed_true = [{0}, {0}, {1}]
    itsets_transposed = io.transpose_context(itsets)
    assert itsets_transposed == itsets_transposed_true
    assert io.transpose_context(io.transpose_context(itsets)) == itsets

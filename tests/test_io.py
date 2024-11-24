import numpy as np
import pandas as pd
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
    datas_transposed = dict(
        pandas=pd.DataFrame([[True, False], [True, False], [False, True]],
                            index=['m1', 'm2', 'm3'], columns=['g1', 'g2']),
        itemset=[{0}, {0}, {1}],
        named_itemset=([{0}, {0}, {1}], ['m1', 'm2', 'm3'], ['g1', 'g2']),
        bitarray=[fbarray('10'), fbarray('10'), fbarray('01')],
        named_bitarray=([fbarray('10'), fbarray('10'), fbarray('01')], ['m1', 'm2', 'm3'], ['g1', 'g2']),
        bool=[[True, False], [True, False], [False, True]],
        named_bool=([[True, False], [True, False], [False, True]], ['m1', 'm2', 'm3'], ['g1', 'g2']),
        dict={'m1': {'g1'}, 'm2': {'g1'}, 'm3': {'g2'}}
    )

    for data_type, data in datas.items():
        transposed = io.transpose_context(data)
        if data_type == 'pandas':
            assert (transposed == datas_transposed[data_type]).all(None), "Problem transposing pd.DataFrame"
            assert (io.transpose_context(transposed) == data).all(None), "Problem transposing pd.DataFrame twice"
            continue

        assert transposed == datas_transposed[data_type], f"Problem transposing {data_type} context"
        assert io.transpose_context(transposed) == data, f"Problem transposing {data_type} context twice"


def test_read_write_cxt():
    data_string = '\n'.join([
        'B', '', '3', '3', '',
        'Consulting', 'Planning', 'Assembly and installation',
        'Furniture', 'Computers', 'Copy machines',
        'XXX', 'XX.', 'XXX', ''
    ])

    data_df = pd.DataFrame(
        [[True, True, True], [True, True, False], [True, True, True]],
        index=['Consulting', 'Planning', 'Assembly and installation'],
        columns=['Furniture', 'Computers', 'Copy machines']
    )
    assert (io.read_cxt(data_string) == data_df).all(None)
    assert io.write_cxt(data_df) == data_string

    with open('test_file.cxt', 'w') as file:
        io.write_cxt(data_df, file)
    with open('test_file.cxt', 'r') as file:
        data_read = io.read_cxt(file)
    assert (data_read == data_df).all(None)
    os.remove('test_file.cxt')


def test_from_fca_repo():
    context_name = 'planets_en'
    assert (io.from_fca_repo(context_name)[0] == io.from_fca_repo(context_name+'.cxt')[0]).all(None)

    context_df = pd.DataFrame([
        [True, False, False, True, False, False, True],  # X..X..X
        [True, False, False, True, False, False, True],  # X..X..X
        [True, False, False, True, False, True, False],  # X..X.X.
        [True, False, False, True, False, True, False],  # X..X.X.
        [False, False, True, False, True, True, False],  # ..X.XX.

        [False, False, True, False, True, True, False],  # ..X.XX.
        [False, True, False, False, True, True, False],  # .X..XX.
        [False, True, False, False, True, True, False],  # .X..XX.
        [True, False, False, False, True, True, False],  # X...XX.
    ],
        index=['Merkur', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto'],
        columns=['Small', 'Medium', 'Large', 'Near', 'Distant', 'Moon', 'No moon']
    )
    metadata = {
        'title': 'Planets',
        'source': "Anggraini, D. (2011). Analisis Perubahan Kelompok Berdasarkan Perubahan Nilai Jual Pada Bloomberg Market Data dengan Menggunakan Formal Concept Analysis, p. 7",
        'size': {'objects': 9, 'attributes': 7},
        'language': 'English',
        'description': 'size and distance of planets'
    }

    context_loaded, meta_loaded = io.from_fca_repo(context_name)
    assert (context_loaded == context_df).all(None)
    assert meta_loaded == metadata


def test_to_mermaid_diagram():
    nodes = ['Top', 'Left Top', 'Left Bottom', 'Right', 'Bottom']
    neighbours = [[1, 3], [2], [4], [4], []]
    diagram_text = '\n'.join([
        'flowchart TD',
        'A["Top"];', 'B["Left Top"];', 'C["Left Bottom"];', 'D["Right"];', 'E["Bottom"];', '',
        'A --- B;', 'A --- D;', 'B --- C;', 'C --- E;', 'D --- E;'
    ])

    txt = io.to_mermaid_diagram(nodes, neighbours)
    assert txt == diagram_text

from dataclasses import dataclass
from typing import Iterable, Iterator, FrozenSet, BinaryIO, Generator, Union
import numpy.typing as npt
import numpy as np
import pandas as pd
from functools import reduce

from bitarray import frozenbitarray as fbarray, bitarray
from bitarray.util import zeros as bazeros


ContextType = Union[pd.DataFrame, dict[str, frozenset[str]], list[set[int]], list[list[bool]], list[fbarray]]


@dataclass
class UnknownContextTypeError(TypeError):
    submitted_type: type

    def __str__(self) -> str:
        return f"Received context is of unknown type: {self.submitted_type}." \
                "Supported types are: pd.DataFrame (with bool values), " \
                    "dictionary with object names as keys and object descriptions as values, " \
                    "list of sets of indices of True-valued columns (so, list of itemsets) " \
                    "list of lists of bool values for every pair of object-attribute " \
                    "list of bitarrays representing objects' descriptions." 


##########################
# Basic type conversions #
##########################
def np2bas(X: npt.ArrayLike) -> list[fbarray]:
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


def to_itemsets(data: ContextType) -> tuple[list[frozenset[int]], list[str], list[str]]:
    """Convert the context defined by `data` into the itemset format

    Parameters
    ----------
    data: ContextType
        Binary formal context in one of the supported formats:
         pd.DataFrame (with bool values);
         dictionary with object names as keys and object descriptions as values;
         list of sets of indices of True-valued columns (so, list of itemsets);
         list of lists of bool values for every pair of object-attribute; or
         list of bitarrays representing objects' descriptions

    Return
    ------
    itemsets: list[frozenset[int]]
        list of sets of indices of True-valued columns
    object_names: list[str]
        names of objects (i.e. rows) in the data.
        If the names are not specified by the data, returns "object_1", "object_2", ...
    attribute_names: list[str]
        names of attributes (i.e. columns) in the data
        If the names are not specified by the data, returns "attribute_1", "attribute_2", ...

    Examples
    --------
    to_itemsets( pd.DataFrame({'a': [False, True], 'b': [True, True]}) )
        --> ([frozenset({1}), frozenset({0, 1})], ['0', '1'], ['a', 'b'])

    to_itemsets( {'row1': ['b'], 'row2': ['a', 'b']} )
        --> ([frozenset({1}), frozenset({0, 1})], ['row1', 'row2'], ['a', 'b'])

    to_itemsets( [[1], [0',1]] )
        --> ([frozenset({1}), frozenset({0, 1})], ['object_0', 'object_1'], ['attribute_0', 'attribute_1'])
    """
    if len(data) == 0:
        return [], [], []
    
    if isinstance(data, pd.DataFrame):
        return list(bas2isets(np2bas(data.values))), list(map(str, data.index)), list(map(str, data.columns))
    
    if isinstance(data, dict):
        objects = sorted(data.keys())
        attributes = sorted(reduce(set.union, data.values(), set()))
        attrs_idx_map = {attr: attr_i for attr_i, attr in enumerate(attributes)}
    
        itemsets = [[attrs_idx_map[attr] for attr in data[obj]] for obj in objects]
        return list(map(frozenset, itemsets)), list(map(str, objects)), list(map(str, attributes))
    
    if isinstance(data, list):
        objects = [f'object_{i}' for i in range(len(data))]        
        
        is_bitarray = data[0] and isinstance(data[0], bitarray)
        is_bool_list = data[0] and isinstance(data[0], (list, tuple)) and isinstance(data[0][0], bool)
        if is_bitarray or is_bool_list:
            attributes = [f'attribute_{j}' for j in range(len(data[0]))]
            itemsets = list(bas2isets(map(bitarray, data)))
            return itemsets, objects, attributes
        
        # if data is given by a list of itemsets
        attributes = sorted(reduce(set.union, data, set()))
        attrs_idx_map = {attr: attr_i for attr_i, attr in enumerate(attributes)}
        itemsets = [[attrs_idx_map[attr] for attr in row] for row in data]
        
        if not isinstance(attributes[0], str):
            attributes = [f'attribute_{j}' for j in range(len(attributes))]

        return list(map(frozenset, itemsets)), list(map(str, objects)), list(map(str, attributes))

    raise UnknownContextTypeError(type(data))


def to_dictionary(data: ContextType) -> dict[str, frozenset[str]]:
    """Convert the context defined by `data` into the dictionary format

    Parameters
    ----------
    data: ContextType
        Binary formal context in one of the supported formats:
         pd.DataFrame (with bool values);
         dictionary with object names as keys and object descriptions as values;
         list of sets of indices of True-valued columns (so, list of itemsets);
         list of lists of bool values for every pair of object-attribute; or
         list of bitarrays representing objects' descriptions

    Return
    ------
    dictionary: dict[str, frozenset[str]]
        dictionary of type 'object_name' -> set of attributes describing object
    object_names: list[str]
        names of objects (i.e. rows) in the data.
        If the names are not specified by the data, returns "object_1", "object_2", ...
    attribute_names: list[str]
        names of attributes (i.e. columns) in the data
        If the names are not specified by the data, returns "attribute_1", "attribute_2", ...

    Examples
    --------
    to_dictionary( pd.DataFrame({'a': [False, True], 'b': [True, True]}) )
        -->  {'0': frozenset({'b'}), '1': frozenset({'a', 'b'})}

    to_dictionary( [bitarray('01'), bitarray('11')] )
        --> {'object_0': frozenset({'attribute_1'}), 'object_1': frozenset({'attribute_0', 'attribute_1'})}

    to_dictionary( [[1], [0,1]] )
        --> {'object_0': frozenset({'attribute_1'}), 'object_1': frozenset({'attribute_0', 'attribute_1'})}
    """
    itemsets, objects, attributes = to_itemsets(data)
    return {obj: frozenset([attributes[attr_i] for attr_i in itemset]) for obj, itemset in zip(objects, itemsets)}


def to_bitarrays(data: ContextType) -> tuple[list[fbarray], list[str], list[str]]:
    """Convert the context defined by `data` into the bitarrays format

    Parameters
    ----------
    data: ContextType
        Binary formal context in one of the supported formats:
         pd.DataFrame (with bool values);
         dictionary with object names as keys and object descriptions as values;
         list of sets of indices of True-valued columns (so, list of itemsets);
         list of lists of bool values for every pair of object-attribute; or
         list of bitarrays representing objects' descriptions

    Return
    ------
    bitarrays: list[frozenbitarray]
        list of frozen bitarrays representing descriptions of objects. For i-th object and j-th attribute,
        `bitarrays[i][j] == True` means that i-th object is described by j-th attribute.
    object_names: list[str]
        names of objects (i.e. rows) in the data.
        If the names are not specified by the data, returns "object_1", "object_2", ...
    attribute_names: list[str]
        names of attributes (i.e. columns) in the data
        If the names are not specified by the data, returns "attribute_1", "attribute_2", ...

    Examples
    --------
    to_bitarrays( pd.DataFrame({'a': [False, True], 'b': [True, True]}) )
        --> ([frozenbitarray('01'), frozenbitarray('11')], ['0', '1'], ['a', 'b'])

    to_bitarrays( {'row1': ['b'], 'row2': ['a','b']} )
        -->  ([frozenbitarray('01'), frozenbitarray('11')], ['row1', 'row2'], ['a', 'b'])

    to_bitarrays( [[1], [0, 1]] )
        --> ([frozenbitarray('01'), frozenbitarray('11')], ['object_0', 'object_1'], ['attribute_0', 'attribute_1'])
    """
    itemsets, objects, attributes = to_itemsets(data)
    return list(isets2bas(itemsets, len(attributes))), objects, attributes
    

def to_pandas(data: ContextType) -> pd.DataFrame:
    """Convert the context defined by `data` into pandas DataFrame format

    Parameters
    ----------
    data: ContextType
        Binary formal context in one of the supported formats:
         pd.DataFrame (with bool values);
         dictionary with object names as keys and object descriptions as values;
         list of sets of indices of True-valued columns (so, list of itemsets);
         list of lists of bool values for every pair of object-attribute; or
         list of bitarrays representing objects' descriptions

    Return
    ------
    dataframe: pd.DataFrame
        Binary pandas.DataFrame representing the `data`

    Examples
    --------
    to_pandas( {'row1': ['b'], 'row2': ['a','b']} )
        --> >           a     b
            > row1  False  True
            > row2   True  True

    to_pandas( [[1], [0, 1]] )
        --> >           attribute_0  attribute_1
            > object_0        False         True
            > object_1         True         True

    to_pandas( [bitarray('01'), bitarray('11')] )
        --> >           attribute_0  attribute_1
            > object_0        False         True
            > object_1         True         True
    """
    itemsets, objects, attributes = to_itemsets(data)
    df = pd.DataFrame(False, index=objects, columns=attributes).fillna(False)
    for obj_i, itemset in enumerate(itemsets):
        df.iloc[obj_i, list(itemset)] = True
    return df
    

def transpose_context(data: ContextType) -> ContextType:
    """Switch places among objects and attributes and transpose their relation

    Parameters
    ----------
    data: ContextType
        Binary formal context in one of the supported formats:
         pd.DataFrame (with bool values);
         dictionary with object names as keys and object descriptions as values;
         list of sets of indices of True-valued columns (so, list of itemsets);
         list of lists of bool values for every pair of object-attribute; or
         list of bitarrays representing objects' descriptions

    Return
    ------
    transposed_data: ContextType
        Transposed context. The format of transposed data is the same as the format of the original data
    """
    if len(data) == 0:
        return data

    if isinstance(data, pd.DataFrame):
        return data.T

    if isinstance(data, dict):
        transposed = {}
        for obj, description in data.items():
            for attr in description:
                if attr not in transposed:
                    transposed[attr] = []
                transposed[attr].append(obj)
        return {k: frozenset(v) for k, v in transposed.items()}

    if isinstance(data, list):
        n_objs = len(data)
        if data[0] and isinstance(data[0], bitarray):
            n_attrs = len(data[0])
            transposed = [bazeros(n_objs) for _ in range(n_attrs)]
            for obj_i, ba in enumerate(data):
                for attr_i in ba.itersearch(True):
                    transposed[attr_i][obj_i] = True
            return list(map(fbarray, transposed))

        if data[0] and isinstance(data[0], (list, tuple)) and isinstance(data[0][0], bool):
            n_attrs = len(data[0])
            transposed = [[False]*n_objs for _ in range(n_attrs)]
            for obj_i, bools in enumerate(data):
                for attr_i, b in enumerate(bools):
                    if b:
                        transposed[attr_i][obj_i] = True
            return transposed

        # if data is given by a list of itemsets
        attributes = sorted(reduce(set.union, data, set()))
        attrs_idx_map = {attr: attr_i for attr_i, attr in enumerate(attributes)}

        transposed = [[] for _ in range(len(attributes))]
        for obj_i, itemset in enumerate(data):
            for attr in itemset:
                attr_i = attrs_idx_map[attr]
                transposed[attr_i].append(obj_i)
        return list(map(frozenset, transposed))

    raise UnknownContextTypeError(type(data))


def verbalise(description: Union[bitarray, Iterable[int]], names: list[str]) -> Iterable[str]:
    """Convert every index i (or every True i-th element of a bitarray) into a human-readable string `names[i]`"""
    if isinstance(description, bitarray):
        return {names[i] for i in description.itersearch(True)}

    if isinstance(description, Generator):
        return (names[i] for i in description)

    return type(description)([names[i] for i in description])


def to_absolute_number(percentage: Union[int, float], total_size: int) -> int:
    """Convert a float percentage into the absolute number of elements w.r.t. `total_size`. Do nothing with integers"""
    if isinstance(percentage, int):
        return percentage
    return int(percentage * total_size)


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


def save_balist(file: BinaryIO, bitarrays: list[bitarray]):
    """Save the list of bitarrays to the binary file (proposed file extension '.bal')"""
    basize = len(bitarrays[0])
    assert all(len(ba) == basize for ba in bitarrays), "All bitarrays should be of the same size"

    file.write(str(basize).encode(encoding='utf-8'))
    file.write(b'n')

    for ba in bitarrays:
        file.write(ba.tobytes())

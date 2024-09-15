from dataclasses import dataclass
from typing import Iterable, Iterator, FrozenSet, BinaryIO, Generator, Union, TextIO, Optional, get_args, Sequence
import numpy.typing as npt
import numpy as np
import pandas as pd
from functools import reduce

from bitarray import frozenbitarray as fbarray, bitarray
from bitarray.util import zeros as bazeros

ItemsetContextType = list[frozenset[int]]  # `j in ctx[i]` shows whether i-th object shares j-th attribute
NamedItemsetContextType = tuple[ItemsetContextType, list[str], list[str]]  # Itemsets + object names + attribute names
BitarrayContextType = list[fbarray]  # `ctx[i][j]` shows whether i-th object shares j-th attribute
NamedBitarrayContextType = tuple[BitarrayContextType, list[str], list[str]]  # bitarrays + object names + attr. names
BoolContextType = list[list[bool]]  # `ctx[i][j]` shows whether i-th object shares j-th attribute
NamedBoolContextType = tuple[BoolContextType, list[str], list[str]]  # bools context + object names + attribute names
DictContextType = dict[str, frozenset[str]]  # `m in ctx[g]` shows whether object g shares attribute m
PandasContextType = pd.DataFrame  # pandas dataframe where row indices are objects and columns indices are attrs.

ContextType = Union[
    PandasContextType,
    ItemsetContextType, NamedItemsetContextType,
    BitarrayContextType, NamedBitarrayContextType,
    BoolContextType, NamedBoolContextType,
    DictContextType
]


@dataclass
class UnknownContextTypeError(TypeError):
    submitted_type: type

    def __str__(self) -> str:
        return f"Received context is of unknown type: {self.submitted_type}. " +\
                f"Supported types are: \n" +\
                '\n'.join([f"* {t}" for t in get_args(ContextType)])


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

    Examples
    --------
    to_itemsets( pd.DataFrame({'a': [False, True], 'b': [True, True]}) )
        --> [frozenset({1}), frozenset({0, 1})]

    to_itemsets( {'row1': ['b'], 'row2': ['a', 'b']} )
        --> [frozenset({1}), frozenset({0, 1})]

    to_itemsets( [[1], [0',1]] )
        --> [frozenset({1}), frozenset({0, 1})]
    """
    return to_named_itemsets(data)[0]


def to_named_itemsets(data: ContextType) -> NamedItemsetContextType:
    """Convert the context defined by `data` into the named itemset format

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
    to_named_itemsets( pd.DataFrame({'a': [False, True], 'b': [True, True]}) )
        --> ([frozenset({1}), frozenset({0, 1})], ['0', '1'], ['a', 'b'])

    to_named_itemsets( {'row1': ['b'], 'row2': ['a', 'b']} )
        --> ([frozenset({1}), frozenset({0, 1})], ['row1', 'row2'], ['a', 'b'])

    to_named_itemsets( [[1], [0',1]] )
        --> ([frozenset({1}), frozenset({0, 1})], ['object_0', 'object_1'], ['attribute_0', 'attribute_1'])
    """
    if len(data) == 0:
        return [], [], []

    is_pandas = isinstance(data, pd.DataFrame)
    is_dict = isinstance(data, dict)\
              and all(isinstance(k, str) for k in data.keys())\
              and all(all(isinstance(v, str) for v in vals) for vals in data.values())
    is_named_sequence = isinstance(data, Sequence) and len(data) == 3 and all(all(isinstance(v, str) for v in names) for names in data[1:])
    crosses_data = data[0] if is_named_sequence else data
    is_sequence = isinstance(crosses_data, Sequence)
    is_bitarrays = is_sequence and isinstance(crosses_data[0], bitarray)
    is_bools = is_sequence and isinstance(crosses_data, (list, tuple)) \
               and all(all(isinstance(v, bool) for v in vals) for vals in crosses_data)
    is_itemsets = is_sequence and not is_bitarrays and not is_bools and isinstance(list(crosses_data[0])[0], int)


    is_supported_type = sum(map(int, [is_pandas, is_dict, is_itemsets, is_bitarrays, is_bools])) == 1
    if not is_supported_type:
        raise UnknownContextTypeError(type(data))

    if is_itemsets and is_named_sequence:
        return data

    # PandasContextType
    if is_pandas:
        return list(bas2isets(np2bas(data.values))), list(map(str, data.index)), list(map(str, data.columns))

    # DictContextType
    if is_dict:
        objects = sorted(data.keys())
        attributes = sorted(reduce(set.union, data.values(), set()))
        attrs_idx_map = {attr: attr_i for attr_i, attr in enumerate(attributes)}

        itemsets = [[attrs_idx_map[attr] for attr in data[obj]] for obj in objects]
        return list(map(frozenset, itemsets)), list(map(str, objects)), list(map(str, attributes))

    # Possible data types: Itemsets or (Named) Bitarrays or (Named) Bools
    objects = data[1] if is_named_sequence else (f'object_{i}' for i in range(len(data)))
    n_attrs = len(crosses_data[0]) if is_bitarrays or is_bools else len(reduce(set.__or__, map(set, crosses_data)))
    attributes = data[2] if is_named_sequence else (f'attribute_{j}' for j in range(n_attrs))
    objects, attributes = list(map(str, objects)), list(map(str, attributes))

    if is_bitarrays or is_bools:
        return list(bas2isets(map(bitarray, crosses_data))), objects, attributes

    # Possible data types: Itemsets
    return list(map(frozenset, crosses_data)), objects, attributes


def to_dictionary(data: ContextType) -> DictContextType:
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
    itemsets, objects, attributes = to_named_itemsets(data)
    return {obj: frozenset([attributes[attr_i] for attr_i in itemset]) for obj, itemset in zip(objects, itemsets)}


def to_named_bitarrays(data: ContextType) -> NamedBitarrayContextType:
    """Convert the context defined by `data` into the named bitarrays format

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
    to_named_bitarrays( pd.DataFrame({'a': [False, True], 'b': [True, True]}) )
        --> ([frozenbitarray('01'), frozenbitarray('11')], ['0', '1'], ['a', 'b'])

    to_named_bitarrays( {'row1': ['b'], 'row2': ['a','b']} )
        -->  ([frozenbitarray('01'), frozenbitarray('11')], ['row1', 'row2'], ['a', 'b'])

    to_named_bitarrays( [[1], [0, 1]] )
        --> ([frozenbitarray('01'), frozenbitarray('11')], ['object_0', 'object_1'], ['attribute_0', 'attribute_1'])
    """
    itemsets, objects, attributes = to_named_itemsets(data)
    return list(isets2bas(itemsets, len(attributes))), objects, attributes


def to_bitarrays(data: ContextType) -> BitarrayContextType:
    """Convert the context defined by `data` into the named bitarrays format

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

    Examples
    --------
    to_bitarrays( pd.DataFrame({'a': [False, True], 'b': [True, True]}) )
        --> [frozenbitarray('01'), frozenbitarray('11')]

    to_bitarrays( {'row1': ['b'], 'row2': ['a','b']} )
        --> [frozenbitarray('01'), frozenbitarray('11')]

    to_bitarrays( [[1], [0, 1]] )
        --> [frozenbitarray('01'), frozenbitarray('11')]
    """
    return to_named_bitarrays(data)[0]


def to_named_bools(data: ContextType) -> NamedBoolContextType:
    """Convert the context defined by `data` into the named bools format

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
    bools: list[list[bool]
        list of lists of bools representing descriptions of objects. For i-th object and j-th attribute,
        `bools[i][j] == True` means that i-th object is described by j-th attribute.
    object_names: list[str]
        names of objects (i.e. rows) in the data.
        If the names are not specified by the data, returns "object_1", "object_2", ...
    attribute_names: list[str]
        names of attributes (i.e. columns) in the data
        If the names are not specified by the data, returns "attribute_1", "attribute_2", ...

    Examples
    --------
    to_named_bools( pd.DataFrame({'a': [False, True], 'b': [True, True]}) )
        --> ([[False, True], [True, True]], ['0', '1'], ['a', 'b'])

    to_named_bools( {'row1': ['b'], 'row2': ['a','b']} )
        -->  ([[False, True], [True, True]], ['row1', 'row2'], ['a', 'b'])

    to_named_bools( [[1], [0, 1]] )
        --> ([[False, True], [True, True]], ['object_0', 'object_1'], ['attribute_0', 'attribute_1'])
    """
    itemsets, objects, attributes = to_named_itemsets(data)
    bools = [[i in itemset for i in range(len(attributes))] for itemset in itemsets]
    return bools, objects, attributes


def to_bools(data: ContextType) -> BoolContextType:
    """Convert the context defined by `data` into the named bools format

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
    bools: list[list[bool]
        list of lists of bools representing descriptions of objects. For i-th object and j-th attribute,
        `bools[i][j] == True` means that i-th object is described by j-th attribute.

    Examples
    --------
    to_bools( pd.DataFrame({'a': [False, True], 'b': [True, True]}) )
        --> [[False, True], [True, True]]

    to_bools( {'row1': ['b'], 'row2': ['a','b']} )
        -->  [[False, True], [True, True]]

    to_bools( [[1], [0, 1]] )
        --> [[False, True], [True, True]]
    """
    return to_named_bools(data)[0]


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
    itemsets, objects, attributes = to_named_itemsets(data)
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


def load_cxt(file: BinaryIO) -> dict[str, set[str]]:
    # TODO: Write the function (or copy from FCApy)
    raise NotImplementedError


def save_cxt(file: BinaryIO, context: ContextType):
    # TODO: Write the function (or copy from FCApy)
    raise NotImplementedError


def from_fca_repo(file_name: str) -> dict[str, set[str]]:
    # TODO: To implement
    raise NotImplementedError


def to_mermaid_diagram(node_labels: list[str], edges: list[tuple[int, int]]) -> str:
    # TODO: Write the function
    raise NotImplementedError







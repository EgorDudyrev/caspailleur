import typing
from dataclasses import dataclass
from itertools import combinations, islice
from string import ascii_uppercase
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


def to_named_itemsets(data: ContextType) -> NamedItemsetContextType:
    """Convert the context defined by `data` into the named itemset format

    Parameters
    ----------
    data: ContextType
        Binary formal context in one of the supported formats:
         pd.DataFrame (with bool values);
         dictionary with object names as keys and named object descriptions as values;
         itemsets where every object description is represented with a set of indices of described attributes;
         bitarrays where every object description is represented with a bitarray where every value corresponds to an attribute;
         list of bools where every object description is represented with a list of bools where every value corresponds to an attribute;

         For itemset, bitarrays, and list of bools types of contexts,
         one can also provide the names of objects and attributes (as shown in the examples).

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

    to_named_itemsets( [[1], [0,1]] )
        --> ([frozenset({1}), frozenset({0, 1})], ['object_0', 'object_1'], ['attribute_0', 'attribute_1'])

    to_named_itemsets( ([[1], [0,1]], ['g1', 'g2'], ['m1', 'm2']) )
        --> ([frozenset({1}), frozenset({0, 1})], ['g1', 'g2'], ['m1', 'm2'])
    """
    if len(data) == 0:
        return [], [], []

    context_type = identify_supported_context_type(data)
    if context_type is None:
        raise UnknownContextTypeError(type(data))

    if context_type == NamedItemsetContextType:
        return data

    if context_type == PandasContextType:
        return list(bas2isets(np2bas(data.values))), list(map(str, data.index)), list(map(str, data.columns))

    if context_type == DictContextType:
        objects = sorted(data.keys())
        attributes = sorted(reduce(set.union, data.values(), set()))
        attrs_idx_map = {attr: attr_i for attr_i, attr in enumerate(attributes)}

        itemsets = [[attrs_idx_map[attr] for attr in data[obj]] for obj in objects]
        return list(map(frozenset, itemsets)), list(map(str, objects)), list(map(str, attributes))

    # Possible data types: Itemsets or (Named) Bitarrays or (Named) Bools
    is_named = context_type in {NamedItemsetContextType, NamedBitarrayContextType, NamedBoolContextType}
    crosses_data = data[0] if is_named else data
    objects = data[1] if is_named else (f'object_{i}' for i in range(len(crosses_data)))

    n_attrs = len(reduce(set.__or__, map(set, crosses_data))) if context_type == ItemsetContextType else len(crosses_data[0])
    attributes = data[2] if is_named else (f'attribute_{j}' for j in range(n_attrs))
    objects, attributes = list(map(str, objects)), list(map(str, attributes))

    if context_type == ItemsetContextType:
        return list(map(frozenset, crosses_data)), objects, attributes

    # Possible data types: (Named) Bitarrays or (Named) Bools
    return list(bas2isets(map(bitarray, crosses_data))), objects, attributes


def to_itemsets(data: ContextType) -> ItemsetContextType:
    """Convert the context defined by `data` into the itemset format

    Parameters
    ----------
    data: ContextType
        Binary formal context in one of the supported formats.
        The list of the supported formats is provided in the Docstring of `io.to_named_itemsets(...)` function.

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


def to_dictionary(data: ContextType) -> DictContextType:
    """Convert the context defined by `data` into the dictionary format

    Parameters
    ----------
    data: ContextType
        Binary formal context in one of the supported formats.
        The list of the supported formats is provided in the Docstring of `io.to_named_itemsets(...)` function.

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
        Binary formal context in one of the supported formats.
        The list of the supported formats is provided in the Docstring of `io.to_named_itemsets(...)` function.

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
        Binary formal context in one of the supported formats.
        The list of the supported formats is provided in the Docstring of `io.to_named_itemsets(...)` function.

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
        Binary formal context in one of the supported formats.
        The list of the supported formats is provided in the Docstring of `io.to_named_itemsets(...)` function.

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
        Binary formal context in one of the supported formats.
        The list of the supported formats is provided in the Docstring of `io.to_named_itemsets(...)` function.

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
        Binary formal context in one of the supported formats.
        The list of the supported formats is provided in the Docstring of `io.to_named_itemsets(...)` function.

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
        Binary formal context in one of the supported formats.
        The list of the supported formats is provided in the Docstring of `io.to_named_itemsets(...)` function.

    Return
    ------
    transposed_data: ContextType
        Transposed context. The format of transposed data is the same as the format of the original data
    """
    if len(data) == 0:
        return data

    data_type = identify_supported_context_type(data)
    if data is None:
        raise UnknownContextTypeError(type(data))

    if data_type == PandasContextType:
        return data.T

    if data_type == DictContextType:
        transposed = {}
        for obj, description in data.items():
            for attr in description:
                if attr not in transposed:
                    transposed[attr] = []
                transposed[attr].append(obj)
        return {k: frozenset(v) for k, v in transposed.items()}

    is_named = data_type in {NamedItemsetContextType, NamedBitarrayContextType, NamedBoolContextType}
    crosses_data = data[0] if is_named else data
    if data_type in {NamedItemsetContextType, ItemsetContextType}:
        n_attrs = len(data[2]) if is_named else len(reduce(set.__or__, crosses_data, set()))
        transposed = [[] for _ in range(n_attrs)]
        for obj_i, itemset in enumerate(crosses_data):
            for attr_i in itemset:
                transposed[attr_i].append(obj_i)
        transposed = list(map(frozenset, transposed))

    if data_type in {NamedBitarrayContextType, BitarrayContextType}:
        n_objs, n_attrs = len(crosses_data), len(crosses_data[0])
        transposed = [bazeros(n_objs) for _ in range(n_attrs)]
        for obj_i, ba in enumerate(crosses_data):
            for attr_i in ba.itersearch(True):
                transposed[attr_i][obj_i] = True
        transposed = list(map(fbarray, transposed))

    if data_type in {NamedBoolContextType, BoolContextType}:
        n_objs, n_attrs = len(crosses_data), len(crosses_data[0])
        transposed = [[False] * n_objs for _ in range(n_attrs)]
        for obj_i, bools in enumerate(crosses_data):
            for attr_i, b in enumerate(bools):
                if b:
                    transposed[attr_i][obj_i] = True

    if is_named:
        return transposed, data[2], data[1]
    return transposed


def identify_supported_context_type(context: ContextType) -> Optional[typing.Type]:
    """Return the supported TypeName of the provided `context`. Return None if the context format is not supported"""
    is_pandas = isinstance(context, pd.DataFrame)
    is_dict = isinstance(context, dict) \
              and all(isinstance(k, str) for k in context.keys()) \
              and all(all(isinstance(v, str) for v in vals) for vals in context.values())
    is_named = isinstance(context, Sequence) and len(context) == 3 and all(
        all(isinstance(v, str) for v in names) for names in context[1:])
    crosses_data = context[0] if is_named else context
    is_sequence = isinstance(crosses_data, Sequence)
    is_bitarrays = is_sequence and isinstance(crosses_data[0], bitarray)
    is_bools = is_sequence and isinstance(crosses_data, (list, tuple)) \
               and all(all(isinstance(v, bool) for v in vals) for vals in crosses_data)
    is_itemsets = is_sequence and not is_bitarrays and not is_bools and isinstance(list(crosses_data[0])[0], int)

    dtypes_selector = [
        (is_pandas, PandasContextType), (is_dict, DictContextType),
        (not is_named and is_itemsets, ItemsetContextType), (is_named and is_itemsets, NamedItemsetContextType),
        (not is_named and is_bitarrays, BitarrayContextType), (is_named and is_bitarrays, NamedBitarrayContextType),
        (not is_named and is_bools, BoolContextType), (is_named and is_bools, NamedBoolContextType),
    ]
    if sum(condition for condition, dtype in dtypes_selector) != 1:
        return None

    return next(dtype for condition, dtype in dtypes_selector if condition)


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


def read_cxt(file: Union[TextIO, str]) -> PandasContextType:
    """Read the file (or string) that describes a formal context in Burmeister format"""
    data = file.read() if not isinstance(file, str) else file

    _, ns, data = data.split('\n\n')
    n_objs, n_attrs = [int(x) for x in ns.split('\n')]

    data = data.strip().split('\n')
    objects, data = data[:n_objs], data[n_objs:]
    attributes, data = data[:n_attrs], data[n_attrs:]

    crosses = [[c == 'X' for c in line] for line in data]
    return to_pandas((crosses, objects, attributes))


def write_cxt(context: ContextType, file: TextIO = None) -> str:
    """Return the formal `context` represented with Burmeister format. Save the string to a `file` if it is provided"""
    crosses, objects, attributes = to_named_itemsets(context)

    file_data = 'B\n\n'
    file_data += f"{len(objects)}\n{len(attributes)}\n"
    file_data += '\n'
    file_data += '\n'.join(objects) + '\n'
    file_data += '\n'.join(attributes) + '\n'

    file_data += '\n'.join([''.join(
        ['X' if idx in crosses_line else '.' for idx in range(len(attributes))])
        for crosses_line in crosses
    ]) + '\n'

    if file:
        file.write(file_data)
    return file_data


def from_fca_repo(context_name: str) -> PandasContextType:
    """Download a formal context from the (git) repository of contexts.

    Go to 'https://fcarepository.org' for more information on the repo.
    """
    import urllib.request

    context_name = context_name + '.cxt' if not context_name.endswith('.cxt') else context_name

    url = f"https://github.com/fcatools/contexts/raw/main/contexts/{context_name}"
    context_data = urllib.request.urlopen(url).read().decode("utf-8")
    return read_cxt(context_data)


def to_mermaid_diagram(node_labels: list[str], neighbours: list[list[int]]) -> str:
    """Create a mermaid flowchart code. The code can be visualised via https://mermaid.live/ or via GitHub markdown"""
    nodes_symbols = (''.join(symbols) for symbols_len in range(1, len(node_labels)+1)
                     for symbols in combinations(ascii_uppercase, symbols_len))
    nodes_symbols = list(islice(nodes_symbols, len(node_labels)))

    nodes_lines = [f'{symbol}["{label}"];' for symbol, label in zip(nodes_symbols, node_labels)]
    edges_lines = [f"{node} --> {nodes_symbols[neighbour]};"
                   for node, neighbours_indices in zip(nodes_symbols, neighbours) for neighbour in neighbours_indices]

    mermaid_lines = ['flowchart TD'] + nodes_lines + [''] + edges_lines
    return '\n'.join(mermaid_lines)

import pytest
import numpy as np

from caspailleur import mine_equivalence_classes as mec
from caspailleur import base_functions as bfunc

from bitarray import frozenbitarray as fbarray
from bitarray.util import zeros as bazeros
from bitarray import bitarray

def test_list_intents_via_LCM():
    itemsets = [
        {0, 3},
        {0, 2},
        {1, 2},
        {1, 2, 3}
    ]

    intents_true = list(bfunc.isets2bas([set(), {0}, {2}, {3}, {0, 2}, {0, 3}, {1, 2}, {1, 2, 3}, {0, 1, 2, 3, 4}], 5))

    intents = mec.list_intents_via_LCM(list(bfunc.isets2bas(itemsets, 5)))
    assert intents == intents_true

    K = np.array([[True, False, False, False, False, True], [False, True, False, False, False, False]])
    itemsets = bfunc.np2bas(K)
    intents_true = list(bfunc.isets2bas([set(), {1}, {0, 5}, {0, 1, 2, 3, 4, 5}], 6))

    intents = mec.list_intents_via_LCM(itemsets)
    assert intents == intents_true

def test_list_intents_via_Lindig():
    itemsets, attr_extents = [bitarray('1001'), bitarray('1010'), bitarray('0110'), bitarray('0111')], [bitarray('1100'), bitarray('0011'), bitarray('0111'), bitarray('1001')] 

    list_data_intents_true = [[],
    [bitarray('1001')],
    [bitarray('1010')],
    [bitarray('0111')],
    [bitarray('1001'), bitarray('1010')],
    [bitarray('1010'), bitarray('0110'), bitarray('0111')],
    [bitarray('1001'), bitarray('0111')],
    [bitarray('1001'), bitarray('1010'), bitarray('0110'), bitarray('0111')],
    [bitarray('0110'), bitarray('0111')]]

    list_data_intents = list_intents_via_Lindig(itemsets, attr_extents)

    assert list_data_intents == list_data_intents_true

def test_list_attribute_concepts():
    intents = [set(), {0}, {2}, {3}, {0, 2}, {0, 3}, {1, 2}, {1, 2, 3},  {0, 1, 2, 3, 4}]
    attr_concepts_true = [1, 6, 2, 3, 8]

    attr_concepts = mec.list_attribute_concepts(list(bfunc.isets2bas(intents, 5)))
    assert attr_concepts == attr_concepts_true


def tests_iter_equivalence_class():
    K = np.array([
        [True, False, False, True, False],
        [True, False, True, False, False],
        [False, True, True, False, False],
        [False, True, True, True, False],
    ])
    attr_extents = list(bfunc.np2bas(K.T))
    eq_class_true = [
        {0, 1, 2, 3, 4}, {0, 1, 2, 3}, {0, 1, 2, 4}, {0, 1, 3, 4}, {0, 2, 3, 4}, {1, 2, 3, 4},
        {0, 1, 2}, {0, 1, 3}, {0, 1, 4}, {0, 2, 3}, {0, 2, 4}, {0, 3, 4}, {1, 2, 4}, {1, 3, 4}, {2, 3, 4},
        {0, 1}, {0, 4}, {1, 4}, {2, 4}, {3, 4}, {4}
    ]

    eq_class = list(mec.iter_equivalence_class(attr_extents, ~fbarray(bazeros(5))))
    assert eq_class == list(bfunc.isets2bas(eq_class_true, 5))


def test_list_keys_via_eqclass():
    eq_class = list(bfunc.isets2bas([
        {0, 1, 2, 3, 4}, {0, 1, 2, 3}, {0, 1, 2, 4}, {0, 1, 3, 4}, {0, 2, 3, 4}, {1, 2, 3, 4},
        {0, 1, 2}, {0, 1, 3}, {0, 1, 4}, {0, 2, 3}, {0, 2, 4}, {0, 3, 4}, {1, 2, 4}, {1, 3, 4}, {2, 3, 4},
        {0, 1}, {0, 4}, {1, 4}, {2, 4}, {3, 4}, {4}
    ], 5))
    keys_true = list(bfunc.isets2bas([{0, 2, 3}, {0, 1}, {4}], 5))

    keys = mec.list_keys_via_eqclass(eq_class)
    assert keys == keys_true


def test_list_passkeys_via_keys():
    keys = [{0, 2, 3}, {0, 1}, {4}, {0, 2, 4}]
    pkeys_true = [{4}]

    pkeys = mec.list_passkeys_via_eqclass(bfunc.isets2bas(keys, 5))
    assert pkeys == list(bfunc.isets2bas(pkeys_true, 5))


def test_list_keys():
    n_attrs = 5
    intents = [set(), {0}, {2}, {3}, {0, 2}, {0, 3}, {1, 2}, {1, 2, 3}, {0, 1, 2, 3, 4}]
    intents = list(bfunc.isets2bas(intents, n_attrs))

    keys_true = [
        (set(), 0),
        ({0}, 1), ({1}, 6), ({2}, 2), ({3}, 3), ({4}, 8),
        ({0, 1}, 8), ({0, 2}, 4), ({0, 3}, 5), ({1, 3}, 7), ({2, 3}, 7),
        ({0, 2, 3}, 8)
    ]
    keys_true = {next(bfunc.isets2bas([key], n_attrs)): intent_i for key, intent_i in keys_true}

    keys = mec.list_keys(intents)
    assert keys == keys_true

    with pytest.raises(AssertionError):
        mec.list_keys(intents[::-1])


def test_list_passkeys():
    n_attrs = 5
    intents = [set(), {0}, {2}, {3}, {0, 2}, {0, 3}, {1, 2}, {1, 2, 3}, {0, 1, 2, 3, 4}]
    intents = list(bfunc.isets2bas(intents, n_attrs))

    pkeys_true = [
        (set(), 0),
        ({0}, 1), ({1}, 6), ({2}, 2), ({3}, 3), ({4}, 8),
        ({0, 2}, 4), ({0, 3}, 5), ({1, 3}, 7), ({2, 3}, 7),
    ]
    pkeys_true = {next(bfunc.isets2bas([key], n_attrs)): intent_i for key, intent_i in pkeys_true}

    pkeys = mec.list_passkeys(intents)
    assert pkeys == pkeys_true

    with pytest.raises(AssertionError):
        mec.list_passkeys(intents[::-1])


def test_list_stable_extents_via_sofia():
    # Data inspired by Animal Movement Context
    all_extents = [
        fbarray('1111111111111111'),  # delta_stab: 8
        fbarray('0000000111111110'),  # delta_stab: 3
        fbarray('0000111101111000'),  # delta_stab: 3
        fbarray('1011111000000000'),  # delta_stab: 3
        fbarray('0000000101111000'),  # delta_stab: 5
        fbarray('0000111000000000'),  # delta_stab: 3
        fbarray('0011000000000000'),  # delta_stab: 2
        fbarray('0000000000000000'),  # delta_stab: 0
    ]

    bin_attributes = [
        fbarray('1011111000000000'),
        fbarray('0000111101111000'),
        fbarray('0000000111111110'),
        fbarray('0011000000000000'),
    ]

    stable_extents = mec.list_stable_extents_via_sofia(bin_attributes, 8)
    assert set(all_extents) == stable_extents

    stable_extents = mec.list_stable_extents_via_sofia(bin_attributes, 8, min_supp=3)
    assert set(all_extents[:-2]) == stable_extents

    stable_extents = mec.list_stable_extents_via_sofia(bin_attributes, 6)
    assert set(all_extents[:-2]) == stable_extents

    stable_extents = mec.list_stable_extents_via_sofia(bin_attributes, 5)
    assert {all_extents[0], all_extents[4]} == stable_extents

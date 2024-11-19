import pytest
import numpy as np

from caspailleur import mine_equivalence_classes as mec
from caspailleur import io

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

    intents_true = list(io.isets2bas([set(), {0}, {2}, {3}, {0, 2}, {0, 3}, {1, 2}, {1, 2, 3}, {0, 1, 2, 3, 4}], 5))

    intents = mec.list_intents_via_LCM(list(io.isets2bas(itemsets, 5)))
    assert intents == intents_true

    freq_intents = mec.list_intents_via_LCM(list(io.isets2bas(itemsets, 5)), min_supp=2)
    freq_intents_true = list(io.isets2bas([set(), {0}, {2}, {3}, {1, 2}], 5))
    assert freq_intents == freq_intents_true
    assert mec.list_intents_via_LCM(list(io.isets2bas(itemsets, 5)), min_supp=0.5) == freq_intents_true

    K = np.array([[True, False, False, False, False, True], [False, True, False, False, False, False]])
    itemsets = io.np2bas(K)
    intents_true = list(io.isets2bas([set(), {1}, {0, 5}, {0, 1, 2, 3, 4, 5}], 6))

    intents = mec.list_intents_via_LCM(itemsets)
    assert intents == intents_true


def test_list_intents_via_lindig():
    K = np.array([
    [True, False, False, True],
    [True, False, True, False],
    [False, True, True, False],
    [False, True, True, True]])

    itemsets = io.np2bas(K)
    attr_extents = io.np2bas(K.T)
    intents_true = ['1111', '1001', '1010', '0111', '1000', '0010', '0001', '0000', '0110']
 
    intents_true = [bitarray(x) for x in intents_true]

    intents = mec.list_intents_via_Lindig(itemsets, attr_extents)
    assert intents == intents_true


def test_list_attribute_concepts():
    intents = [set(), {0}, {2}, {3}, {0, 2}, {0, 3}, {1, 2}, {1, 2, 3},  {0, 1, 2, 3, 4}]
    attr_concepts_true = [1, 6, 2, 3, 8]

    attr_concepts = mec.list_attribute_concepts(list(io.isets2bas(intents, 5)))
    assert attr_concepts == attr_concepts_true


def tests_iter_equivalence_class():
    K = np.array([
        [True, False, False, True, False],
        [True, False, True, False, False],
        [False, True, True, False, False],
        [False, True, True, True, False],
    ])
    attr_extents = list(io.np2bas(K.T))
    eq_class_true = [
        {0, 1, 2, 3, 4}, {0, 1, 2, 3}, {0, 1, 2, 4}, {0, 1, 3, 4}, {0, 2, 3, 4}, {1, 2, 3, 4},
        {0, 1, 2}, {0, 1, 3}, {0, 1, 4}, {0, 2, 3}, {0, 2, 4}, {0, 3, 4}, {1, 2, 4}, {1, 3, 4}, {2, 3, 4},
        {0, 1}, {0, 4}, {1, 4}, {2, 4}, {3, 4}, {4}
    ]

    eq_class = list(mec.iter_equivalence_class(attr_extents, ~fbarray(bazeros(5))))
    assert eq_class == list(io.isets2bas(eq_class_true, 5))

    eq_class = list(mec.iter_equivalence_class(attr_extents))
    assert eq_class == list(io.isets2bas(eq_class_true, 5))


def test_iter_equivalence_class():
    K = np.array([
        [True, False, False, True, False],
        [True, False, True, False, False],
        [False, True, True, False, False],
        [False, True, True, True, False],
    ])
    attr_extents = list(io.np2bas(K.T))
    eq_class_true = [
        {0, 1, 2, 3, 4}, {0, 1, 2, 3}, {0, 1, 2, 4}, {0, 1, 3, 4}, {0, 2, 3, 4}, {1, 2, 3, 4},
        {0, 1, 2}, {0, 1, 3}, {0, 1, 4}, {0, 2, 3}, {0, 2, 4}, {0, 3, 4}, {1, 2, 4}, {1, 3, 4}, {2, 3, 4},
        {0, 1}, {0, 4}, {1, 4}, {2, 4}, {3, 4}, {4}
    ]

    eq_class = list(mec.iter_equivalence_class_levelwise(attr_extents, ~fbarray(bazeros(5))))
    assert eq_class == list(io.isets2bas(eq_class_true, 5))

    eq_class = list(mec.iter_equivalence_class_levelwise(attr_extents))
    assert eq_class == list(io.isets2bas(eq_class_true, 5))

    eq_class = list(mec.iter_equivalence_class_levelwise(attr_extents, presort_output=False))
    assert set(eq_class) == set(io.isets2bas(eq_class_true, 5))


def test_list_keys_via_eqclass():
    eq_class = list(io.isets2bas([
        {0, 1, 2, 3, 4}, {0, 1, 2, 3}, {0, 1, 2, 4}, {0, 1, 3, 4}, {0, 2, 3, 4}, {1, 2, 3, 4},
        {0, 1, 2}, {0, 1, 3}, {0, 1, 4}, {0, 2, 3}, {0, 2, 4}, {0, 3, 4}, {1, 2, 4}, {1, 3, 4}, {2, 3, 4},
        {0, 1}, {0, 4}, {1, 4}, {2, 4}, {3, 4}, {4}
    ], 5))
    keys_true = list(io.isets2bas([{0, 2, 3}, {0, 1}, {4}], 5))

    keys = mec.list_keys_via_eqclass(eq_class)
    assert keys == keys_true


def test_list_passkeys_via_keys():
    keys = [{0, 2, 3}, {0, 1}, {4}, {0, 2, 4}]
    pkeys_true = [{4}]

    pkeys = mec.list_passkeys_via_eqclass(io.isets2bas(keys, 5))
    assert pkeys == list(io.isets2bas(pkeys_true, 5))


def test_iter_keys_of_intent():
    n_attrs = 5
    intents = [set(), {0}, {2}, {3}, {0, 2}, {0, 3}, {1, 2}, {1, 2, 3}, {0, 1, 2, 3, 4}]
    intents = list(io.isets2bas(intents, n_attrs))

    attr_extents = [{0, 1}, {2, 3}, {0, 2, 3}, {1, 2}, set()]
    attr_extents = list(io.isets2bas(attr_extents, n_attrs))

    keys_true = [
        [set()],  # intent: set()
        [{0}],  # intent: {0}
        [{2}],  # intent: {2}
        [{3}],  # intent: {3}
        [{0, 2}],  # intent: {0, 2}
        [{0, 3}],  # intent: {0, 3}
        [{1}],  # intent: {1, 2}
        [{1, 3}, {2, 3}],  # intent: {1, 2, 3}
        [{4}, {0, 1}, {0, 2, 3}],  # intent: {0, 1, 2, 3, 4}
    ]
    keys_true = [list(io.isets2bas(keys, n_attrs)) for keys in keys_true]

    for key_list_true, intent in zip(keys_true, intents):
        key_list = list(mec.iter_keys_of_intent(intent, attr_extents))
        assert set(key_list) == set(key_list_true), f'Problematic intent: {intent}'


def test_iter_keys_of_intent_pretentious():
    n_attrs = 5
    intents = [set(), {0}, {2}, {3}, {0, 2}, {0, 3}, {1, 2}, {1, 2, 3}, {0, 1, 2, 3, 4}]
    intents = list(io.isets2bas(intents, n_attrs))

    attr_extents = [{0, 1}, {2, 3}, {0, 2, 3}, {1, 2}, set()]
    attr_extents = list(io.isets2bas(attr_extents, n_attrs))

    keys_true = [
        [set()],  # intent: set()
        [{0}],  # intent: {0}
        [{2}],  # intent: {2}
        [{3}],  # intent: {3}
        [{0, 2}],  # intent: {0, 2}
        [{0, 3}],  # intent: {0, 3}
        [{1}],  # intent: {1, 2}
        [{1, 3}, {2, 3}],  # intent: {1, 2, 3}
        [{4}, {0, 1}, {0, 2, 3}],  # intent: {0, 1, 2, 3, 4}
    ]
    keys_true = [list(io.isets2bas(keys, n_attrs)) for keys in keys_true]

    for key_list_true, intent in zip(keys_true, intents):
        key_list = list(mec.iter_keys_of_intent_pretentious(intent, attr_extents))
        assert set(key_list) == set(key_list_true), f'Problematic intent: {intent}'


def test_list_keys():
    n_attrs = 5
    intents = [set(), {0}, {2}, {3}, {0, 2}, {0, 3}, {1, 2}, {1, 2, 3}, {0, 1, 2, 3, 4}]
    intents = list(io.isets2bas(intents, n_attrs))

    keys_true = [
        (set(), 0),
        ({0}, 1), ({1}, 6), ({2}, 2), ({3}, 3), ({4}, 8),
        ({0, 1}, 8), ({0, 2}, 4), ({0, 3}, 5), ({1, 3}, 7), ({2, 3}, 7),
        ({0, 2, 3}, 8)
    ]
    keys_true = {next(io.isets2bas([key], n_attrs)): intent_i for key, intent_i in keys_true}

    keys = mec.list_keys(intents)
    assert keys == keys_true

    with pytest.raises(AssertionError):
        mec.list_keys(intents[::-1])


def test_list_passkeys():
    n_attrs = 5
    intents = [set(), {0}, {2}, {3}, {0, 2}, {0, 3}, {1, 2}, {1, 2, 3}, {0, 1, 2, 3, 4}]
    intents = list(io.isets2bas(intents, n_attrs))

    pkeys_true = [
        (set(), 0),
        ({0}, 1), ({1}, 6), ({2}, 2), ({3}, 3), ({4}, 8),
        ({0, 2}, 4), ({0, 3}, 5), ({1, 3}, 7), ({2, 3}, 7),
    ]
    pkeys_true = {next(io.isets2bas([key], n_attrs)): intent_i for key, intent_i in pkeys_true}

    pkeys = mec.list_passkeys(intents)
    assert pkeys == pkeys_true

    with pytest.raises(AssertionError):
        mec.list_passkeys(intents[::-1])


def test_list_keys_for_extents():
    attr_extents = [{0, 1}, {2, 3}, {0, 2, 3}, {1, 2}, set()]
    extents = [{0, 1, 2, 3},
               {0, 1}, {0, 2, 3}, {1, 2},
               {0}, {1}, {2, 3}, {2}, set()]
    keys_true = [
        (set(), 0),
        ({0}, 1), ({1}, 6), ({2}, 2), ({3}, 3), ({4}, 8),
        ({0, 1}, 8), ({0, 2}, 4), ({0, 3}, 5), ({1, 3}, 7), ({2, 3}, 7),
        ({0, 2, 3}, 8)
    ]

    n_attrs, n_objs = len(attr_extents), len(extents[0])
    attr_extents, extents = [list(io.isets2bas(isets, n_objs)) for isets in [attr_extents, extents]]
    keys_true = {next(io.isets2bas([key], n_attrs)): intent_i for key, intent_i in keys_true}

    keys = mec.list_keys_for_extents(extents, attr_extents)
    assert keys == keys_true

    # Partially Ordered Set case
    attr_extents = [{0, 1, 2}, {0, 1, 3}, {0}, {0}, {1}]
    extents = [{0, 1, 2}, {0, 1, 3}, {0}, {1}, set()]
    keys_true = [
        ({0}, 0), ({1}, 1), ({2}, 2), ({3}, 2),
        ({4}, 3), ({2, 4}, 4), ({3, 4}, 4)
    ]
    n_attrs, n_objs = len(attr_extents), 4
    attr_extents, extents = [list(io.isets2bas(isets, n_objs)) for isets in [attr_extents, extents]]
    keys_true = {next(io.isets2bas([key], n_attrs)): intent_i for key, intent_i in keys_true}

    keys = mec.list_keys_for_extents(extents, attr_extents)
    assert keys == keys_true


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


def test_list_stable_extents_via_gsofia():
    attr_extents = [fbarray('110'),  fbarray('101'), fbarray('011')]
    extents_true = {
        fbarray('111'),
        fbarray('110'), fbarray('101'), fbarray('011'),
        fbarray('100'), fbarray('010'), fbarray('001'),
        fbarray('000')
    }

    stable_extents = mec.list_stable_extents_via_gsofia(attr_extents, min_delta_stability=0)
    assert stable_extents == extents_true

    stable_extents = mec.list_stable_extents_via_gsofia(attr_extents, min_delta_stability=1)
    assert stable_extents == extents_true - {fbarray('000')}

    stable_extents = mec.list_stable_extents_via_gsofia(attr_extents, min_delta_stability=2)
    assert stable_extents == set()

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
    stable_extents = mec.list_stable_extents_via_gsofia(bin_attributes, min_delta_stability=0)
    assert set(all_extents) == stable_extents

    stable_extents = mec.list_stable_extents_via_gsofia(bin_attributes, min_delta_stability=3)
    assert set(all_extents[:-2]) == stable_extents

    stable_extents = mec.list_stable_extents_via_gsofia(bin_attributes, min_delta_stability=4)
    assert {all_extents[0], all_extents[4]} == stable_extents

    stable_extents = mec.list_stable_extents_via_gsofia(bin_attributes, min_delta_stability=0.25)
    assert {all_extents[0], all_extents[4]} == stable_extents

    stable_extents = mec.list_stable_extents_via_gsofia(bin_attributes, min_delta_stability=3, min_supp=6)
    assert {all_extents[0], all_extents[1], all_extents[2], all_extents[3]} == stable_extents

    stable_extents = mec.list_stable_extents_via_gsofia(bin_attributes,
                                                        min_delta_stability=3, min_supp=5, n_stable_extents=2)
    assert {all_extents[0], all_extents[4]} == stable_extents

    stable_extents = mec.list_stable_extents_via_gsofia(bin_attributes, n_stable_extents=3)
    assert {all_extents[0], all_extents[4]} == stable_extents


def test_iter_minimal_rare_itemsets_via_mrgexp():
    # using example from the Towards Minimal Rare Itemset paper
    attr_extents = [fbarray('11101'), fbarray('10111'), fbarray('01111'), fbarray('10000'), fbarray('10111')]
    mris_true = [fbarray('11100'), fbarray('10101'), fbarray('00010')]

    mris = list(mec.iter_minimal_rare_itemsets_via_mrgexp(attr_extents, 2))
    assert set(mris) == set(mris_true)


def test_generate_next_level_descriptions():
    descrs = [tuple()]
    next_descriptions_true = ((0,), (1,), (2,))
    next_descriptions, _ = zip(*mec.generate_next_level_descriptions(descrs, None, 3))
    assert next_descriptions == next_descriptions_true

    descrs = [(1, 2), (2, 3), (1, 3), (1, 4)]
    next_descriptions_true = ((1, 2, 3),)
    next_descriptions, _ = zip(*mec.generate_next_level_descriptions(descrs, None, 3))
    assert next_descriptions == next_descriptions_true

    attr_extents = [fbarray('00000'), fbarray('11101'), fbarray('11011'), fbarray('10011'), fbarray('11111')]
    supports_true = (2,)
    next_descriptions, supports = zip(*mec.generate_next_level_descriptions(descrs, attr_extents, None))
    assert next_descriptions == next_descriptions_true
    assert supports == supports_true


def test_iter_minimal_broad_clusterings_via_mrgexp():
    # using inverse example from the Towards Minimal Rare Itemset paper
    attr_extents = [fbarray('00010'), fbarray('01000'), fbarray('10000'), fbarray('01111'), fbarray('01000')]
    clusterings_true = [fbarray('11100'), fbarray('10101'), fbarray('00010')]

    clusterings = list(mec.iter_minimal_broad_clusterings_via_mrgexp(attr_extents, 3))
    assert set(clusterings) == set(clusterings_true)

    clusterings = list(mec.iter_minimal_broad_clusterings_via_mrgexp(attr_extents, 3, min_added_coverage=4))
    clusterings_true = [fbarray('00010')]
    assert clusterings == clusterings_true

    clusterings = list(mec.iter_minimal_broad_clusterings_via_mrgexp(attr_extents, 3, min_added_coverage=5))
    clusterings_true = []
    assert clusterings == clusterings_true

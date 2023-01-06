import pytest
import numpy as np

from caspailleur import mine_equivalence_classes as mec
from caspailleur import order
from caspailleur import base_functions as bfunc


def test_list_intents_via_LCM():
    itemsets = [
        {0, 3},
        {0, 2},
        {1, 2},
        {1, 2, 3}
    ]
    intents_true = [{0, 1, 2, 3, 4}, {1, 2, 3}, {0, 2}, {0, 3}, {1, 2}, {0}, {2}, {3}, set()]
    intents = mec.list_intents_via_LCM(itemsets, n_attrs=5)
    assert intents == intents_true

    K = np.array([[True, False, False, False, False, True], [False, True, False, False, False, False]])
    itemsets = bfunc.np2isets(K)
    intents_true = [{0, 1, 2, 3, 4, 5}, {0, 5}, {1}, set()]
    intents = mec.list_intents_via_LCM(itemsets)
    assert intents == intents_true


def test_list_attribute_concepts():
    intents = [{0, 1, 2, 3, 4}, {1, 2, 3}, {0, 2}, {0, 3}, {1, 2}, {0}, {2}, {3}, set()]
    attr_concepts_true = [5, 4, 6, 7, 0]

    intents_ba = [bfunc.iset2ba(intent, length=5) for intent in intents]
    parents = order.sort_ba_subsumption(intents_ba)
    attr_concepts = mec.list_attribute_concepts(intents_ba, parents)

    assert attr_concepts == attr_concepts_true


def tests_iter_equivalence_class():
    K = np.array([
        [True, False, False, True, False],
        [True, False, True, False, False],
        [False, True, True, False, False],
        [False, True, True, True, False],
    ])
    attr_extents = list(bfunc.iter_attribute_extents(K))
    eq_class_true = [
        {0, 1, 2, 3, 4}, {0, 1, 2, 3}, {0, 1, 2, 4}, {0, 1, 3, 4}, {0, 2, 3, 4}, {1, 2, 3, 4},
        {0, 1, 2}, {0, 1, 3}, {0, 1, 4}, {0, 2, 3}, {0, 2, 4}, {0, 3, 4}, {1, 2, 4}, {1, 3, 4}, {2, 3, 4},
        {0, 1}, {0, 4}, {1, 4}, {2, 4}, {3, 4}, {4}
    ]

    eq_class = list(mec.iter_equivalence_class(attr_extents, [0, 1, 2, 3, 4]))
    assert eq_class == eq_class_true


def test_list_keys_via_eqclass():
    eq_class = (D for D in [
        {0, 1, 2, 3, 4}, {0, 1, 2, 3}, {0, 1, 2, 4}, {0, 1, 3, 4}, {0, 2, 3, 4}, {1, 2, 3, 4},
        {0, 1, 2}, {0, 1, 3}, {0, 1, 4}, {0, 2, 3}, {0, 2, 4}, {0, 3, 4}, {1, 2, 4}, {1, 3, 4}, {2, 3, 4},
        {0, 1}, {0, 4}, {1, 4}, {2, 4}, {3, 4}, {4}
    ])
    keys_true = [{0, 2, 3}, {0, 1}, {4}]

    keys = mec.list_keys_via_eqclass(eq_class)
    assert keys == keys_true


def test_list_passkeys_via_keys():
    keys = [{0, 2, 3}, {0, 1}, {4}]
    pkeys_true = [{4}]

    pkeys = mec.list_passkeys_via_keys(keys)
    assert pkeys == pkeys_true


def test_list_keys():
    intents = [set(), {0}, {2}, {3}, {0, 2}, {0, 3}, {1, 2}, {1, 2, 3}, {0, 1, 2, 3, 4}]

    keys_true = [
        (set(), 0),
        ({0}, 1),
        ({1}, 6),
        ({2}, 2),
        ({3}, 3),
        ({4}, 8),
        ({0, 1}, 8),
        ({0, 2}, 4),
        ({0, 3}, 5),
        ({1, 3}, 7),
        ({2, 3}, 7),
        ({0, 2, 3}, 8)
    ]
    keys_true = {frozenset(key): intent_i for key, intent_i in keys_true}

    keys = mec.list_keys(intents)
    assert keys == keys_true

    with pytest.raises(AssertionError):
        mec.list_keys(intents[::-1])

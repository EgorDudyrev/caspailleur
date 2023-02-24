from caspailleur import implication_bases as impbas
from caspailleur import mine_equivalence_classes as mec
from caspailleur import base_functions as bfunc

from bitarray import frozenbitarray as fbarray


def test_list_proper_premises_via_keys():
    n_attrs = 5
    intents = [set(), {0}, {2}, {3}, {0, 2}, {0, 3}, {1, 2}, {1, 2, 3}, {0, 1, 2, 3, 4}]
    intents = [bfunc.iset2ba(iset, n_attrs) for iset in intents]
    keys_to_intents = mec.list_keys(intents)

    pprems_true = [bfunc.iset2ba(pp, 5) for pp in [{1}, {4}, {2, 3}, {0, 1}, {0, 2, 3}]]

    pprems = impbas.iter_proper_premises_via_keys(intents, keys_to_intents)

    assert set(pprems) == set(pprems_true)


def test_list_pseudo_intents_via_keys():
    pintents_true = [{1}, {0, 1, 2}, {2, 3}, {4}]
    intents = [set(), {0}, {2}, {3}, {0, 2}, {0, 3}, {1, 2}, {1, 2, 3}, {0, 1, 2, 3, 4}]
    keys_intent_map = [
        ([], 0), ([0], 1), ([1], 6), ([2], 2), ([3], 3), ([4], 8),
        ([0, 1], 8), ([0, 2], 4), ([0, 3], 5), ([1, 3], 7), ([2, 3], 7), ([0, 2, 3], 8)
    ]
    keys_intent_map = dict([(frozenset(k), v) for k, v in keys_intent_map])

    intents = [bfunc.iset2ba(iset, 5) for iset in intents]
    pintents_true = [bfunc.iset2ba(iset, 5) for iset in pintents_true]
    keys_intent_map = {bfunc.iset2ba(k, 5): v for k, v in keys_intent_map.items()}

    pintents = impbas.list_pseudo_intents_via_keys(keys_intent_map.items(), intents)
    assert {pintent for pintent, _ in pintents} == set(pintents_true)

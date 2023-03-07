import numpy as np

from caspailleur.orchestrator import explore_data
from caspailleur.indices import linearity_index, distributivity_index
from caspailleur.base_functions import isets2bas


def test_explore_data():
    K = np.array([
        [True, False, False, True, False],
        [True, False, True, False, False],
        [False, True, True, False, False],
        [False, True, True, True, False],
    ])

    intents_true = [set(), {0}, {2}, {3}, {0, 2}, {0, 3}, {1, 2}, {1, 2, 3}, {0, 1, 2, 3, 4}]
    intents_true = [frozenset(intent) for intent in intents_true]

    keys_true = [
        (set(), 0),
        ({0}, 1), ({1}, 6), ({2}, 2), ({3}, 3), ({4}, 8),
        ({0, 1}, 8), ({0, 2}, 4), ({0, 3}, 5), ({1, 3}, 7), ({2, 3}, 7),
        ({0, 2, 3}, 8)
    ]
    keys_true = {frozenset(key): intent_i for key, intent_i in keys_true}

    passkeys_true = [
        (set(), 0),
        ({0}, 1), ({1}, 6), ({2}, 2), ({3}, 3), ({4}, 8),
        ({0, 2}, 4), ({0, 3}, 5), ({1, 3}, 7), ({2, 3}, 7),
    ]
    passkeys_true = {frozenset(key): intent_i for key, intent_i in passkeys_true}

    pseudo_intents_true = [({4}, 8), ({1}, 6), ({2, 3}, 7), ({0, 1, 2}, 8)]
    pseudo_intents_true = {frozenset(pi): intent_i for pi, intent_i in pseudo_intents_true}

    proper_premises_true = [({1}, 6), ({4}, 8), ({0, 1}, 8), ({2, 3}, 7), ({0, 2, 3}, 8)]
    proper_premises_true = {frozenset(pp): intent_i for pp, intent_i in proper_premises_true}

    parents_ordering_true = [set(), {0}, {0}, {0}, {1, 2}, {1, 3}, {2}, {3, 6}, {4, 5, 7}]
    transitive_parents = [set(), {0}, {0}, {0}, {0, 1, 2}, {0, 1, 3}, {0, 2}, {0, 2, 3, 6}, {0, 1, 2, 3, 4, 5, 6, 7}]
    n_trans_parents = sum(len(tpars) for tpars in transitive_parents)
    linearity_true = linearity_index(n_trans_parents, len(intents_true))
    distributivity_true = distributivity_index(
        list(isets2bas(intents_true, K.shape[1])),
        list(isets2bas(parents_ordering_true, len(intents_true))),
        n_trans_parents
    )

    explore_data_true = dict(
        intents=intents_true, keys=keys_true, passkeys=passkeys_true,
        pseudo_intents=pseudo_intents_true, proper_premises=proper_premises_true,
        intents_ordering=parents_ordering_true, linearity=linearity_true, distributivity=distributivity_true
    )

    explored_data = explore_data(K)
    assert explored_data['intents'] == intents_true
    assert explored_data['keys'] == keys_true
    assert explored_data['passkeys'] == passkeys_true
    assert explored_data['pseudo_intents'] == pseudo_intents_true
    assert explored_data['proper_premises'] == proper_premises_true
    assert explored_data['intents_ordering'] == parents_ordering_true
    assert explored_data['linearity'] == linearity_true
    assert explored_data['distributivity'] == distributivity_true

    assert explored_data == explore_data_true

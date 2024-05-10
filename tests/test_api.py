import pandas as pd

from caspailleur import api


def test_iter_descriptions():
    data = {'g1': ['a', 'b'], 'g2': ['b', 'c']}

    descriptions_data_true = [
        {
            'description': set(), 'extent': {'g1', 'g2'}, 'closure': {'b'},
            'support': 2, 'delta-stability': 0,
            'is_closed': False, 'is_key': True, 'is_passkey': True, 'is_proper_premise': True, 'is_pseudo_intent': True
        }, {
            'description': {'a'}, 'extent': {'g1'}, 'closure': {'a', 'b'},
            'support': 1, 'delta-stability': 0,
            'is_closed': False, 'is_key': True, 'is_passkey': True, 'is_proper_premise': False, 'is_pseudo_intent': False
        }, {
            'description': {'b'}, 'extent': {'g1', 'g2'}, 'closure': {'b'},
            'support': 2, 'delta-stability': 1,
            'is_closed': True, 'is_key': False, 'is_passkey': False, 'is_proper_premise': False, 'is_pseudo_intent': False
        }, {
            'description': {'c'}, 'extent': {'g2'}, 'closure': {'b', 'c'},
            'support': 1, 'delta-stability': 0,
            'is_closed': False, 'is_key': True, 'is_passkey': True, 'is_proper_premise': False, 'is_pseudo_intent': False
        }, {
            'description': {'a', 'b'}, 'extent': {'g1'}, 'closure': {'a', 'b'},
            'support': 1, 'delta-stability': 1,
            'is_closed': True, 'is_key': False, 'is_passkey': False, 'is_proper_premise': False, 'is_pseudo_intent': False
        }, {
            'description': {'a', 'c'}, 'extent': set(), 'closure': {'a', 'b', 'c'},
            'support': 0, 'delta-stability': 0,
            'is_closed': False, 'is_key': True, 'is_passkey': True, 'is_proper_premise': False, 'is_pseudo_intent': False
        }, {
            'description': {'b', 'c'}, 'extent': {'g2'}, 'closure': {'b', 'c'},
            'support': 1, 'delta-stability': 1,
            'is_closed': True, 'is_key': False, 'is_passkey': False, 'is_proper_premise': False, 'is_pseudo_intent': False
        }, {
            'description': {'a', 'b', 'c'}, 'extent': set(), 'closure': {'a', 'b', 'c'},
            'support': 0, 'delta-stability': 0,
            'is_closed': True, 'is_key': False, 'is_passkey': False, 'is_proper_premise': False, 'is_pseudo_intent': False
        },
    ]
    descriptions_data = list(api.iter_descriptions(data))
    assert descriptions_data == descriptions_data_true


def test_mine_descriptions():
    data = {'g1': ['a', 'b'], 'g2': ['b', 'c']}

    descriptions_data_true = pd.DataFrame({
        'description': [set(), {'a'}, {'b'}, {'c'}, {'a', 'b'}, {'a', 'c'}, {'b', 'c'}, {'a', 'b', 'c'}],
        'extent': [{'g1', 'g2'}, {'g1'}, {'g1', 'g2'}, {'g2'}, {'g1'}, set(), {'g2'}, set()],
        'closure': [{'b'}, {'a', 'b'}, {'b'}, {'b', 'c'}, {'a', 'b'}, {'a', 'b', 'c'}, {'b', 'c'}, {'a', 'b', 'c'}],
        'support': [2, 1, 2, 1, 1, 0, 1, 0],
        'delta-stability': [0, 0, 1, 0, 1, 0, 1, 0],
        'is_closed': [False, False, True, False, True, False, True, True],
        'is_key': [True, True, False, True, False, True, False, False],
        'is_passkey': [True, True, False, True, False, True, False, False],
        'is_proper_premise': [True, False, False, False, False, False, False, False],
        'is_pseudo_intent': [True, False, False, False, False, False, False, False]
    })
    descriptions_data = api.mine_descriptions(data)
    assert list(descriptions_data.index) == list(descriptions_data_true.index)
    assert list(descriptions_data.columns) == list(descriptions_data_true.columns)
    assert (descriptions_data == descriptions_data_true).all(None)

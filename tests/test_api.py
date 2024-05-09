import pandas as pd

from caspailleur import api


def test_iter_descriptions():
    data = {'g1': ['a','b'], 'g2': ['b', 'c']}

    descriptions_data_true = [
        {'description': set(), 'extent': {'g1','g2'}, 'support': 2},
        {'description': {'a'}, 'extent': {'g1'}, 'support': 1},
        {'description': {'b'}, 'extent': {'g1','g2'}, 'support': 2},
        {'description': {'c'}, 'extent': {'g2'}, 'support': 1},
        {'description': {'a', 'b'}, 'extent': {'g1'}, 'support': 1},
        {'description': {'a', 'c'}, 'extent': set(), 'support': 0},
        {'description': {'b', 'c'}, 'extent': {'g2'}, 'support': 1},
        {'description': {'a','b','c'}, 'extent': set(), 'support': 0},
    ]
    descriptions_data = list(api.iter_descriptions(data))
    assert descriptions_data == descriptions_data_true


def test_mine_descriptions():
    data = {'g1': ['a','b'], 'g2': ['b', 'c']}

    descriptions_data_true = pd.DataFrame({
        'description': [set(), {'a'}, {'b'}, {'c'}, {'a', 'b'}, {'a','c'}, {'b','c'}, {'a','b','c'}],
        'extent': [{'g1','g2'}, {'g1'}, {'g1','g2'}, {'g2'}, {'g1'}, set(), {'g2'}, set()],
        'support': [2, 1, 2, 1, 1, 0, 1, 0],
    })
    descriptions_data = api.mine_descriptions(data)
    assert (descriptions_data == descriptions_data_true).all(None)

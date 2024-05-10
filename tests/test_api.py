import pandas as pd

from caspailleur import api


def test_mine_descriptions():
    data = {'g1': ['a', 'b'], 'g2': ['b', 'c']}

    descriptions_data_true = pd.DataFrame({
        'description': [set(), {'a'}, {'b'}, {'c'}, {'a', 'b'}, {'a', 'c'}, {'b', 'c'}, {'a', 'b', 'c'}],
        'extent': [{'g1', 'g2'}, {'g1'}, {'g1', 'g2'}, {'g2'}, {'g1'}, set(), {'g2'}, set()],
        'intent': [{'b'}, {'a', 'b'}, {'b'}, {'b', 'c'}, {'a', 'b'}, {'a', 'b', 'c'}, {'b', 'c'}, {'a', 'b', 'c'}],
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
    for f in descriptions_data:
        assert (descriptions_data[f] == descriptions_data_true[f]).all(), f"Problematic column {f}"


def test_iter_descriptions():
    data = {'g1': ['a', 'b'], 'g2': ['b', 'c']}

    descriptions_data = list(api.iter_descriptions(data))
    assert (pd.DataFrame(descriptions_data) == api.mine_descriptions(data)).all(None)


def test_mine_concepts():
    data = {'g1': ['a', 'b'], 'g2': ['b', 'c']}

    concepts_df_true = pd.DataFrame({
        'extent': [{'g1', 'g2'}, {'g1'}, {'g2'}, set()],
        'intent': [{'b'}, {'a', 'b'}, {'b', 'c'}, {'a', 'b', 'c'}],
        'support': [2, 1, 1, 0],
        'delta-stability': [1, 1, 1, 0],
        'keys': [[set()], [{'a'}], [{'c'}], [{'a','c'}]],
        'passkeys': [[set()], [{'a'}], [{'c'}], [{'a', 'c'}]],
        'proper_premises': [[set()], [], [], []],
        'pseudo_intents': [[set()], [], [], []]
    })

    concepts_df = api.mine_concepts(data)
    assert list(concepts_df.index) == list(concepts_df_true.index)
    assert list(concepts_df.columns) == list(concepts_df_true.columns)
    for f in concepts_df:
        assert (concepts_df[f] == concepts_df_true[f]).all(), f'Problematic column {f}'


def test_mine_implications():
    data = {'g1': ['a', 'b'], 'g2': ['b', 'c']}

    impls_df_true = pd.DataFrame({
        'premise': [set()],
        'conclusion': [{'b'}],
        'conclusion_full': [{'b'}],
        'extent': [{'g1', 'g2'}],
        'support': 2
    })

    impls_df = api.mine_implications(data, 'proper premise')
    assert list(impls_df.index) == list(impls_df_true.index)
    assert list(impls_df.columns) == list(impls_df_true.columns)
    for f in impls_df:
        assert (impls_df[f] == impls_df_true[f]).all(), f'Problematic column {f}'

    impls_df = api.mine_implications(data, 'pseudo-intent')
    assert list(impls_df.index) == list(impls_df_true.index)
    assert list(impls_df.columns) == list(impls_df_true.columns)
    for f in impls_df:
        assert (impls_df[f] == impls_df_true[f]).all(), f'Problematic column {f}'

import pandas as pd

from caspailleur import api


def assert_df_equality(df1: pd.DataFrame, df2: pd.DataFrame):
    assert list(df1.index) == list(df2.index)
    assert list(df1.columns) == list(df2.columns)
    for f in df1:
        assert list(df1[f]) == list(df2[f]), f"Problematic column: {f}"


def test_mine_descriptions():
    data = {'g1': ['a', 'b'], 'g2': ['b', 'c']}

    descriptions_data_true = pd.DataFrame({
        'description': [set(), {'a'}, {'b'}, {'c'}, {'a', 'b'}, {'a', 'c'}, {'b', 'c'}, {'a', 'b', 'c'}],
        'extent': [{'g1', 'g2'}, {'g1'}, {'g1', 'g2'}, {'g2'}, {'g1'}, set(), {'g2'}, set()],
        'intent': [{'b'}, {'a', 'b'}, {'b'}, {'b', 'c'}, {'a', 'b'}, {'a', 'b', 'c'}, {'b', 'c'}, {'a', 'b', 'c'}],
        'support': [2, 1, 2, 1, 1, 0, 1, 0],
        'delta_stability': [0, 0, 1, 0, 1, 0, 1, 0],
        'is_closed': [False, False, True, False, True, False, True, True],
        'is_key': [True, True, False, True, False, True, False, False],
        'is_passkey': [True, True, False, True, False, True, False, False],
        'is_proper_premise': [True, False, False, False, False, False, False, False],
        'is_pseudo_intent': [True, False, False, False, False, False, False, False]
    })
    descriptions_data = api.mine_descriptions(data, to_compute='all')
    assert_df_equality(descriptions_data, descriptions_data_true)

    # test min_support threshold
    for min_supp in [1, 2]:
        freq_df_true = descriptions_data_true[descriptions_data_true['support'] >= min_supp].reset_index(drop=True)
        freq_df = api.mine_descriptions(data, min_support=min_supp, to_compute='all')
        assert_df_equality(freq_df, freq_df_true)

    for min_supp in [0.5, 1.0]:
        freq_df_true = descriptions_data_true[descriptions_data_true['support']/len(data) >= min_supp].reset_index(drop=True)
        freq_df = api.mine_descriptions(data, min_support=min_supp, to_compute='all')
        assert_df_equality(freq_df, freq_df_true)


def test_iter_descriptions():
    data = {'g1': ['a', 'b'], 'g2': ['b', 'c']}

    descriptions_data = list(api.iter_descriptions(data))
    assert_df_equality(pd.DataFrame(descriptions_data), api.mine_descriptions(data))

    to_compute = ['description', 'extent', 'intent', 'is_proper_premise', 'is_pseudo_intent', 'delta_stability']
    descriptions_data = list(api.iter_descriptions(data, to_compute=to_compute))
    assert_df_equality(pd.DataFrame(descriptions_data), api.mine_descriptions(data, to_compute='all')[to_compute])


def test_mine_concepts():
    data = {'g1': ['a', 'b'], 'g2': ['b', 'c']}

    concepts_df_true = pd.DataFrame({
        'extent': [{'g1', 'g2'}, {'g1'}, {'g2'}, set()],
        'intent': [{'b'}, {'a', 'b'}, {'b', 'c'}, {'a', 'b', 'c'}],
        'support': [2, 1, 1, 0],
        'delta_stability': [1, 1, 1, 0],
        'keys': [[set()], [{'a'}], [{'c'}], [{'a','c'}]],
        'passkeys': [[set()], [{'a'}], [{'c'}], [{'a', 'c'}]],
        'proper_premises': [[set()], [], [], []],
        'pseudo_intents': [[set()], [], [], []],
        'previous_concepts': [{1, 2}, {3}, {3}, set()],
        'next_concepts': [set(), {0}, {0}, {1, 2}],
        'sub_concepts': [{1, 2, 3}, {3}, {3}, set()],
        'super_concepts': [set(), {0}, {0}, {0, 1, 2}],
    })

    concepts_df = api.mine_concepts(data, to_compute='all')
    assert_df_equality(concepts_df, concepts_df_true)

    stable_concepts_df_true = concepts_df_true[:3]
    for f in ['previous_concepts', 'next_concepts', 'sub_concepts', 'super_concepts']:
        stable_concepts_df_true[f] = stable_concepts_df_true[f] - {3}

    stable_concepts_df = api.mine_concepts(data, to_compute='all', min_delta_stability=1)
    assert_df_equality(stable_concepts_df, stable_concepts_df_true)

    stable_concepts_df = api.mine_concepts(data, to_compute='all', n_stable_concepts=3)
    assert_df_equality(stable_concepts_df, stable_concepts_df_true)

    stable_concepts_df = api.mine_concepts(data, to_compute='all', min_delta_stability=1, n_stable_concepts=3)
    assert_df_equality(stable_concepts_df, stable_concepts_df_true)


def test_mine_implications():
    data = {'g1': ['a', 'b'], 'g2': ['b', 'c']}

    impls_df_true = pd.DataFrame({
        'premise': [set()],
        'conclusion': [{'b'}],
        'conclusion_full': [{'b'}],
        'extent': [{'g1', 'g2'}],
        'support': 2
    })

    impls_df = api.mine_implications(data, 'Proper Premise')
    assert_df_equality(impls_df, impls_df_true)

    impls_df = api.mine_implications(data, 'Pseudo-Intent')
    assert_df_equality(impls_df, impls_df_true)

    for basis_name in ['Duquenne-Guigues', 'Minimum', 'Canonical']:
        impls_df = api.mine_implications(data, basis_name)
        assert_df_equality(impls_df, impls_df_true)

    for basis_name in ['Canonical Direct', 'Karell']:
        impls_df = api.mine_implications(data, basis_name)
        assert_df_equality(impls_df, impls_df_true)

    impls_df_true_unit = pd.DataFrame({
        'premise': [set()],
        'conclusion': ['b'],
        'conclusion_full': [{'b'}],
        'extent': [{'g1', 'g2'}],
        'support': 2
    })
    impls_df = api.mine_implications(data, 'Proper Premise', unit_base=True)
    assert_df_equality(impls_df, impls_df_true_unit)

    impls_df = api.mine_implications(data, 'Proper Premise', unit_base=True,
                                     to_compute=['premise', 'conclusion', 'extent'])
    assert_df_equality(impls_df, impls_df_true_unit[['premise', 'conclusion', 'extent']])

    # TODO: Add refined tests for implication (and especially unit) base

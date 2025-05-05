import pandas as pd
import caspailleur as csp


def test_data_load():
    df, meta = csp.io.from_fca_repo('famous_animals_en')
    df_true = pd.DataFrame({
        'cartoon': [True, True, False, False, False],
        'real': [False, False, True, True, True],
        'tortoise': [False, False, False, False, True],
        'dog': [False, True, False, True, False],
        'cat': [True, False, True, False, False],
        'mammal': [True, True, True, True, False]
    }, index=['Garfield', 'Snoopy', 'Socks', "Greyfriar's Bobby", "Harriet"])

    meta_true = {
        'title': 'Famous Animals',
        'source': 'Priss, U. (2006), Formal concept analysis in information science. Ann. Rev. Info. Sci. Tech., 40: 521-543. p.525',
        'size': {'objects': 5, 'attributes': 6},
        'language': 'English',
        'description': 'famous animals and their characteristics'
    }
    assert (df == df_true).all(None)
    assert meta == meta_true


def test_mining_concepts():
    df = pd.DataFrame({
        'cartoon': [True, True, False, False, False],
        'real': [False, False, True, True, True],
        'tortoise': [False, False, False, False, True],
        'dog': [False, True, False, True, False],
        'cat': [True, False, True, False, False],
        'mammal': [True, True, True, True, False]
    }, index=['Garfield', 'Snoopy', 'Socks', "Greyfriar's Bobby", "Harriet"])
    concepts_df = csp.mine_concepts(df)

    extents_true = [
        "Garfield, Greyfriar's Bobby, Harriet, Snoopy, Socks", "Garfield, Greyfriar's Bobby, Snoopy, Socks",
        "Greyfriar's Bobby, Harriet, Socks", 'Garfield, Snoopy', "Greyfriar's Bobby, Socks",
        "Greyfriar's Bobby, Snoopy", 'Garfield, Socks', 'Harriet', 'Snoopy', 'Garfield', "Greyfriar's Bobby", 'Socks',
        ''
    ]

    intents_true = [
        '', 'mammal', 'real', 'cartoon, mammal', 'mammal, real', 'dog, mammal', 'cat, mammal', 'real, tortoise',
        'cartoon, dog, mammal', 'cartoon, cat, mammal', 'dog, mammal, real', 'cat, mammal, real',
        'cartoon, cat, dog, mammal, real, tortoise'
    ]

    assert concepts_df['extent'].map(sorted).map(', '.join).tolist() == extents_true
    assert concepts_df['intent'].map(sorted).map(', '.join).tolist() == intents_true

    concepts_df = csp.mine_concepts(
        df, min_support=3, min_delta_stability=1,
        to_compute=['intent', 'keys', 'support', 'delta_stability', 'sub_concepts']
    )
    intents_true = [set(), {'mammal'}, {'real'}]
    keys_true = [[set()], [{'mammal'}], [{'real'}]]
    supports_true = [5, 4, 3]
    delta_stability_true = [1, 2, 1]
    sub_concepts_true = [{1, 2}, set(), set()]
    assert concepts_df['intent'].tolist() == intents_true
    assert concepts_df['keys'].tolist() == keys_true
    assert concepts_df['support'].tolist() == supports_true
    assert concepts_df['delta_stability'].tolist() == delta_stability_true
    assert concepts_df['sub_concepts'].tolist() == sub_concepts_true


def test_mining_implications():
    df = pd.DataFrame({
        'cartoon': [True, True, False, False, False],
        'real': [False, False, True, True, True],
        'tortoise': [False, False, False, False, True],
        'dog': [False, True, False, True, False],
        'cat': [True, False, True, False, False],
        'mammal': [True, True, True, True, False]
    }, index=['Garfield', 'Snoopy', 'Socks', "Greyfriar's Bobby", "Harriet"])
    implications_df = csp.mine_implications(df)

    premises_true = [
        'cartoon', 'tortoise', 'dog', 'cat',
    ]
    assert implications_df['premise'].map(sorted).map(', '.join).tolist() == premises_true
    conclusions_true = [
        'mammal', 'real', 'mammal', 'mammal',
    ]
    assert implications_df['conclusion'].map(sorted).map(', '.join).tolist() == conclusions_true
    supports_true = [2, 1, 2, 2]
    assert implications_df['support'].tolist() == supports_true

    implications_df = csp.mine_implications(
        df, basis_name='Canonical', unit_base=True,
        to_compute=['premise', 'conclusion', 'extent'],
        min_support=2
    )
    assert implications_df['premise'].map(set).tolist() == [{'cat'}, {'dog'}, {'cartoon'}]
    assert implications_df['conclusion'].tolist() == ['mammal', 'mammal', 'mammal']
    assert implications_df['extent'].tolist() == [{'Garfield', 'Socks'}, {"Greyfriar's Bobby", 'Snoopy'},
                                                  {'Garfield', 'Snoopy'}]


def test_mining_descriptions():
    df = pd.DataFrame({
        'cartoon': [True, True, False, False, False],
        'real': [False, False, True, True, True],
        'tortoise': [False, False, False, False, True],
        'dog': [False, True, False, True, False],
        'cat': [True, False, True, False, False],
        'mammal': [True, True, True, True, False]
    }, index=['Garfield', 'Snoopy', 'Socks', "Greyfriar's Bobby", "Harriet"])

    descriptions_df = csp.mine_descriptions(df)
    assert df.shape[1] == 6
    assert len(descriptions_df) == 64
    assert ', '.join(descriptions_df.columns) == \
           'description, extent, intent, support, delta_stability, ' \
           'is_closed, is_key, is_passkey, is_proper_premise, is_pseudo_intent'

    assert descriptions_df['description'].tolist()[:3] == [set(), {'cartoon'}, {'real'}]
    assert descriptions_df['support'].tolist()[:3] == [5, 2, 3]
    assert descriptions_df['is_key'].tolist()[:3] == [True, True, True]


def test_visualising_lattice():
    df = pd.DataFrame({
        'cartoon': [True, True, False, False, False],
        'real': [False, False, True, True, True],
        'tortoise': [False, False, False, False, True],
        'dog': [False, True, False, True, False],
        'cat': [True, False, True, False, False],
        'mammal': [True, True, True, True, False]
    }, index=['Garfield', 'Snoopy', 'Socks', "Greyfriar's Bobby", "Harriet"])

    concepts_df = csp.mine_concepts(df, min_support=2)

    new_intent_labels = ('<b>' + concepts_df['new_intent'].map(sorted).map(', '.join) + '</b>').replace('<b></b>', '')
    old_intent_labels = (concepts_df['intent'] - concepts_df['new_intent']).map(sorted).map(', '.join)
    intent_labels = (new_intent_labels + ';' + old_intent_labels).map(lambda l: ', '.join(l.strip(';').split(';')))
    extent_labels = concepts_df['extent'].map(sorted).map(', '.join)

    node_labels = (intent_labels + '<br><br>' + extent_labels)
    node_labels = node_labels.replace(' ', '&nbsp;')

    diagram_code = csp.io.to_mermaid_diagram(node_labels, concepts_df['previous_concepts'])

    diagram_code_true = 'flowchart TD\nA["<br><br>Garfield, Greyfriar\'s Bobby, Harriet, Snoopy, Socks"];\nB["<b>mammal</b><br><br>Garfield, Greyfriar\'s Bobby, Snoopy, Socks"];\nC["<b>real</b><br><br>Greyfriar\'s Bobby, Harriet, Socks"];\nD["<b>cartoon</b>, mammal<br><br>Garfield, Snoopy"];\nE["mammal, real<br><br>Greyfriar\'s Bobby, Socks"];\nF["<b>dog</b>, mammal<br><br>Greyfriar\'s Bobby, Snoopy"];\nG["<b>cat</b>, mammal<br><br>Garfield, Socks"];\n\nA --- B;\nA --- C;\nB --- D;\nB --- E;\nB --- F;\nB --- G;\nC --- E;'
    assert diagram_code == diagram_code_true


def test_data_convertion():
    df = pd.DataFrame({
        'cartoon': [True, True, False, False, False],
        'real': [False, False, True, True, True],
        'tortoise': [False, False, False, False, True],
        'dog': [False, True, False, True, False],
        'cat': [True, False, True, False, False],
        'mammal': [True, True, True, True, False]
    }, index=['Garfield', 'Snoopy', 'Socks', "Greyfriar's Bobby", "Harriet"])

    assert (csp.io.to_pandas(df) == df).all(None)
    assert csp.io.to_itemsets(df) == [{0, 4, 5}, {0, 3, 5}, {1, 4, 5}, {1, 3, 5}, {1, 2}]
    assert csp.io.to_named_itemsets(df) == (
        [{0, 4, 5}, {0, 3, 5}, {1, 4, 5}, {1, 3, 5}, {1, 2}],
        ['Garfield', 'Snoopy', 'Socks', "Greyfriar's Bobby", 'Harriet'],
        ['cartoon', 'real', 'tortoise', 'dog', 'cat', 'mammal']
    )
    from bitarray import bitarray
    assert csp.io.to_bitarrays(df) == [
        bitarray('100011'), bitarray('100101'), bitarray('010011'),
        bitarray('010101'), bitarray('011000')
    ]

    assert csp.io.to_named_bitarrays(df) == (
        [bitarray('100011'), bitarray('100101'), bitarray('010011'),
        bitarray('010101'), bitarray('011000')],
        ['Garfield', 'Snoopy', 'Socks', "Greyfriar's Bobby", 'Harriet'],
        ['cartoon', 'real', 'tortoise', 'dog', 'cat', 'mammal']
    )

    assert csp.io.to_bools(df) == [
        [True, False, False, False, True, True],
        [True, False, False, True, False, True],
        [False, True, False, False, True, True],
        [False, True, False, True, False, True],
        [False, True, True, False, False, False],
    ]

    assert csp.io.to_named_bools(df) == (
        [[True, False, False, False, True, True], [True, False, False, True, False, True],
         [False, True, False, False, True, True], [False, True, False, True, False, True],
         [False, True, True, False, False, False]],
        ['Garfield', 'Snoopy', 'Socks', "Greyfriar's Bobby", 'Harriet'],
        ['cartoon', 'real', 'tortoise', 'dog', 'cat', 'mammal']
    )

    assert csp.io.to_dictionary(df) == {
        'Garfield': {'cartoon', 'mammal', 'cat'},
        'Snoopy': {'dog', 'cartoon', 'mammal'},
        'Socks': {'real', 'mammal', 'cat'},
        "Greyfriar's Bobby": {'real', 'mammal', 'dog'},
        'Harriet': {'real', 'tortoise'}
    }

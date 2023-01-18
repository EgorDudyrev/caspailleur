import pytest

from caspailleur.base_functions import iset2ba
from caspailleur import order


def test_topological_sorting():
    elements = [iset2ba(iset, 3) for iset in [[0, 1, 2], [0, 1], [1], [2], []]]
    assert order.topological_sorting(elements) == (elements, [0, 1, 2, 3, 4])

    idxs_unordered = [1, 0, 3, 2, 4]
    elements_unordered = [elements[i] for i in idxs_unordered]
    orig_to_topsort_map_true = [idxs_unordered.index(i) for i in range(len(idxs_unordered))]
    assert order.topological_sorting(elements_unordered) == (elements, orig_to_topsort_map_true)


def test_construct_spanning_tree():
    elements = [iset2ba(iset, 3) for iset in [[0, 1, 2], [0, 1], [1], [2], []]]
    sptree_true = [None, 0, 1, 0, 3]
    sptree = order.construct_spanning_tree(elements)
    assert sptree == sptree_true


def test_split_sptree_to_chains():
    # elements = [iset2ba(iset, 3) for iset in [[0, 1, 2], [0, 1], [1], [2], []]]
    sptree = [None, 0, 1, 0, 3]
    chains_true = [[0, 1, 2], [0, 3, 4]]

    chains = order.split_sptree_to_chains(sptree)
    assert chains == chains_true


def test_find_parents_in_chains():
    elements = [iset2ba(iset, 3) for iset in [[0, 1, 2], [0, 1], [1], [2], []]]
    chains = [[0, 1, 2], [0, 3, 4]]
    parents_per_chains_true = [[], [0, 0], [1, 0], [0, 0], [2, 3]]  # for now, we include the transitive parents
    parents_true = [set(parents) for parents in parents_per_chains_true]

    parents_list = order.find_parents_in_chains(chains, elements, use_tqdm=False)
    assert parents_list == parents_true


def test_drop_transitive_parents():
    parents_per_chains = [set(), {0}, {0, 1}, {0}, {2, 3}]
    parents_nontrans_true = [set(), {0}, {1}, {0}, {2, 3}]

    parents_nontrans = order.drop_transitive_parents(parents_per_chains)
    assert parents_nontrans == parents_nontrans_true


def test_sort_ba_inclusion():
    elements = [iset2ba(iset, 3) for iset in [[0, 1, 2], [0, 1], [1], [2], []]]
    inclusion_order_true = [set(), {0}, {1}, {0}, {2, 3}]
    inclusion_order = order.sort_ba_inclusion(elements, use_tqdm=False)
    assert inclusion_order == inclusion_order_true

    elements_unordered = [[0, 1], [0, 1, 2], [2], [1], []]
    elements_unordered = [iset2ba(iset, 3) for iset in elements_unordered]
    order_unordered =    [{1},    set(),     {1}, {0}, {2, 3}]
    assert order.sort_ba_inclusion(elements_unordered) == order_unordered


def test_inverse_order():
    inclusion_order = [set(), {0}, {1}, {0}, {2, 3}]
    subsumption_order_true = [{1, 3}, {2}, {4}, {4}, set()]
    assert order.inverse_order(inclusion_order) == subsumption_order_true
    assert order.inverse_order(order.inverse_order(inclusion_order)) == inclusion_order


def test_sort_ba_subsumption():
    elements = [iset2ba(iset, 3) for iset in [[0, 1, 2], [0, 1], [1], [2], []]]
    subsumption_order_true = [{1, 3}, {2}, {4}, {4}, set()]
    assert order.sort_ba_subsumption(elements) == subsumption_order_true

    elements_unordered = [iset2ba(iset, 3) for iset in [[0, 1], [0, 1, 2], [2], [1], []]]
    order_unordered = [{3}, {0, 2}, {4}, {4}, set()]
    assert order.sort_ba_subsumption(elements_unordered) == order_unordered


def test_sort_intents_by_inclusion():
    elements = [set(), {1}, {2}, {0, 1}, {0, 1, 2}]
    inclusion_order_true = [{1, 2}, {3}, {4}, {4}, set()]
    inclusion_order = order.sort_intents_inclusion(elements)
    assert inclusion_order == inclusion_order_true

    with pytest.raises(AssertionError):
        order.sort_intents_inclusion(elements[::-1])

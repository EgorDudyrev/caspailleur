import pytest

from caspailleur import order, base_functions as bfunc


def test_topological_sorting():
    elements = [frozenset(x) for x in [[], [1], [2], [0, 1], [0, 1, 2]]]
    elements_ba = list(bfunc.isets2bas(elements, 5))
    assert order.topological_sorting(elements_ba) == (elements_ba, [0, 1, 2, 3, 4])

    idxs_unordered = [1, 0, 3, 2, 4]
    elements_unordered = [elements_ba[i] for i in idxs_unordered]
    orig_to_topsort_map_true = [idxs_unordered.index(i) for i in range(len(idxs_unordered))]
    assert order.topological_sorting(elements_unordered) == (elements_ba, orig_to_topsort_map_true)


def test_inverse_order():
    inclusion_order = [set(), {0}, {1}, {0}, {2, 3}]
    subsumption_order_true = [{1, 3}, {2}, {4}, {4}, set()]

    inclusion_order_ba = list(bfunc.isets2bas(inclusion_order, 5))
    subsumption_order_true_ba = list(bfunc.isets2bas(subsumption_order_true, 5))

    assert order.inverse_order(inclusion_order_ba) == subsumption_order_true_ba
    assert order.inverse_order(order.inverse_order(inclusion_order_ba)) == inclusion_order_ba


def test_sort_intents_by_inclusion():
    elements = [set(), {1}, {2}, {0, 1}, {0, 1, 2}]
    elements_ba = list(bfunc.isets2bas(elements, 3))

    inclusion_order_true = [{1, 2}, {3}, {4}, {4}, set()]
    inclusion_order_true_ba = list(bfunc.isets2bas(inclusion_order_true, 5))

    inclusion_order = order.sort_intents_inclusion(elements_ba)
    assert inclusion_order == inclusion_order_true_ba

    with pytest.raises(AssertionError):
        order.sort_intents_inclusion(elements_ba[::-1])

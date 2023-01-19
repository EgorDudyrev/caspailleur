import pytest

from caspailleur import order


def test_topological_sorting():
    elements = [frozenset(x) for x in [[], [1], [2], [0, 1], [0, 1, 2]]]
    assert order.topological_sorting(elements) == (elements, [0, 1, 2, 3, 4])

    idxs_unordered = [1, 0, 3, 2, 4]
    elements_unordered = [elements[i] for i in idxs_unordered]
    orig_to_topsort_map_true = [idxs_unordered.index(i) for i in range(len(idxs_unordered))]
    assert order.topological_sorting(elements_unordered) == (elements, orig_to_topsort_map_true)


def test_inverse_order():
    inclusion_order = [set(), {0}, {1}, {0}, {2, 3}]
    subsumption_order_true = [{1, 3}, {2}, {4}, {4}, set()]
    assert order.inverse_order(inclusion_order) == subsumption_order_true
    assert order.inverse_order(order.inverse_order(inclusion_order)) == inclusion_order


def test_sort_intents_by_inclusion():
    elements = [set(), {1}, {2}, {0, 1}, {0, 1, 2}]
    inclusion_order_true = [{1, 2}, {3}, {4}, {4}, set()]
    inclusion_order = order.sort_intents_inclusion(elements)
    assert inclusion_order == inclusion_order_true

    with pytest.raises(AssertionError):
        order.sort_intents_inclusion(elements[::-1])

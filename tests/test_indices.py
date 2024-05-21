from bitarray import frozenbitarray as fbarray

from caspailleur import indices as idxs


def test_delta_stability_by_description():
    attr_extents = [
        fbarray('100101'),
        fbarray('011010'),
        fbarray('100101'),
        fbarray('010010'),
        fbarray('001000'),
        fbarray('111101'),
        fbarray('000010')
    ]

    delta_stab = idxs.delta_stability_by_description(set(), attr_extents)
    assert delta_stab == 1

    delta_stab = idxs.delta_stability_by_description(fbarray('0'*7), attr_extents)
    assert delta_stab == 1

    delta_stab = idxs.delta_stability_by_description([5], attr_extents)
    assert delta_stab == 2
    delta_stab = idxs.delta_stability_by_description(fbarray('0000010'), attr_extents)
    assert delta_stab == 2

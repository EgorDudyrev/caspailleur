from caspailleur import implication_bases as impbas
from caspailleur import mine_equivalence_classes as mec
from caspailleur import base_functions as bfunc

from bitarray import frozenbitarray as fbarray


def test_list_proper_premises_via_lattice():
    n_attrs = 5
    intents = [{0, 1, 2, 3, 4}, {1, 2, 3}, {0, 2}, {0, 3}, {1, 2}, {0}, {2}, {3}, set()][::-1]
    intents_ba = [bfunc.iset2ba(iset, n_attrs) for iset in intents]
    keys_to_intents = mec.list_keys(intents_ba)
    keys_to_intents = {frozenset(bfunc.ba2iset(k)): v for k, v in keys_to_intents.items()}

    pprems_true = [frozenset(pp) for pp in [{1}, {4}, {2, 3}, {0, 1}, {0, 2, 3}]]

    pprems = impbas.iter_proper_premises_via_keys(intents, keys_to_intents)

    assert set(pprems) == set(pprems_true)


def test_list_pseudo_intents_incremental():
    attr_extents = [{0, 1}, {2, 3}, {1, 2, 3}, {0, 3}, set()]
    pintents_true = [{1}, {0, 1, 2}, {2, 3}, {4}]
    intents = [{0, 1, 2, 3, 4}, {1, 2, 3}, {0, 2}, {0, 3}, {1, 2}, {0}, {2}, {3}, set()][::-1]
    pintents = impbas.list_pseudo_intents_incremental(attr_extents, intents)
    assert pintents == pintents_true

from caspailleur import implication_bases as impbas
from caspailleur import mine_equivalence_classes as mec
from caspailleur import base_functions as bfunc

from bitarray import frozenbitarray as fbarray


def test_list_proper_premises_via_lattice():
    intents = [{0, 1, 2, 3, 4}, {1, 2, 3}, {0, 2}, {0, 3}, {1, 2}, {0}, {2}, {3}, set()]
    N_OBJS, N_ATTRS = 4, 5
    intents_ba = [bfunc.iset2ba(intent, N_ATTRS) for intent in intents]
    attr_extents = [bfunc.iset2ba(ext, N_OBJS) for ext in [{0, 1}, {2, 3}, {1, 2, 3}, {0, 3}, set()]]
    keys_to_intents = {
        bfunc.iset2ba(key, N_ATTRS): intent_i
        for intent_i, intent in enumerate(intents)
        for key in mec.list_keys_via_eqclass(mec.iter_equivalence_class(attr_extents, intent))
    }

    pprems_true = [{1}, {4}, {2, 3}, {0, 1}, {0, 2, 3}]
    pprems_true = [bfunc.iset2ba(pprem, N_ATTRS) for pprem in pprems_true]

    pprems = impbas.iter_proper_premises_via_keys(intents_ba, keys_to_intents)

    assert set(pprems) == set(pprems_true)


def test_list_pseudo_intents_incremental():
    attr_extents = [{0, 1}, {2, 3}, {1, 2, 3}, {0, 3}, set()]
    pintents_true = [{1}, {0, 1, 2}, {2, 3}, {4}]
    intents = [{0, 1, 2, 3, 4}, {1, 2, 3}, {0, 2}, {0, 3}, {1, 2}, {0}, {2}, {3}, set()][::-1]
    pintents = impbas.list_pseudo_intents_incremental(attr_extents, intents)
    assert pintents == pintents_true

from typing import Dict, Any

import numpy as np

from . import base_functions as bfuncs
from . import implication_bases as ibases
from . import mine_equivalence_classes as mec
from . import order as ordermod
from . import indices as indicesmod


def explore_data(K: np.ndarray, min_sup: float = 1) -> Dict[str, Any]:
    itemsets = bfuncs.np2isets(K)
    n_attrs = K.shape[1]
    attr_extents = [frozenset(bfuncs.ba2iset(ext)) for ext in bfuncs.iter_attribute_extents(K)]

    intents = mec.list_intents_via_LCM(itemsets, min_supp=min_sup, n_attrs=n_attrs)
    keys = mec.list_keys(intents)
    passkeys = mec.list_passkeys(intents)
    pseudo_intents = ibases.list_pseudo_intents_incremental(attr_extents, intents)

    children_ordering = ordermod.sort_intents_inclusion(intents)
    parents_ordering = ordermod.inverse_order(children_ordering)
    transitive_parents = ordermod.trans_close_relation(parents_ordering)
    linearity = indicesmod.linearity_index(transitive_parents)

    proper_premises = list(ibases.iter_proper_premises_via_keys(intents, keys))
    distributivity = indicesmod.distributivity_index(intents, transitive_parents)

    return dict(
        intents=intents, keys=keys, passkeys=passkeys,
        pseudo_intents=pseudo_intents, proper_premises=proper_premises,
        intents_ordering=parents_ordering, linearity=linearity, distributivity=distributivity
    )

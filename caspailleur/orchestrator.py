from typing import Dict, Any

import numpy as np

from . import base_functions as bfuncs
from . import implication_bases as ibases
from . import mine_equivalence_classes as mec
from . import order as ordermod
from . import indices as indicesmod


def explore_data(K: np.ndarray, min_sup: float = 1) -> Dict[str, Any]:
    itemsets = bfuncs.np2isets(K)
    attr_extents = [frozenset(bfuncs.ba2iset(ext)) for ext in bfuncs.iter_attribute_extents(K)]

    intents = mec.list_intents_via_LCM(itemsets, min_supp=min_sup)[::-1]
    keys = mec.list_keys(intents)
    passkeys = mec.list_passkeys(intents)
    pseudo_intents = ibases.list_pseudo_intents_incremental(attr_extents, intents)

    ordering = ordermod.sort_intents_inclusion(intents)
    linearity = indicesmod.linearity_index(ordering, )

    n_attrs = K.shape[1]
    intents_ba = [bfuncs.iset2ba(intent, n_attrs) for intent in intents]
    keys_ba = {bfuncs.iset2ba(key, n_attrs): intent_i for key, intent_i in keys.items()}
    proper_premises = ibases.iter_proper_premises_via_keys(intents_ba, keys_ba)
    distributivity = indicesmod.distributivity_index(intents_ba)

    return dict(
        intents=intents, keys=keys, passkeys=passkeys,
        pseudo_intents=pseudo_intents, proper_premises=proper_premises,
        intents_ordering=ordering, linearity=linearity, distributivity=distributivity
    )

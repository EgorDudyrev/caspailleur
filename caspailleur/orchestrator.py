from typing import Dict, Any

import numpy as np

from . import base_functions as bfuncs
from . import implication_bases as ibases
from . import mine_equivalence_classes as mec
from . import order as ordermod
from . import indices as indicesmod


def explore_data(K: np.ndarray, min_sup: float = 1, return_itemsets: bool = True) -> Dict[str, Any]:
    itemsets = list(bfuncs.np2bas(K))

    intents = mec.list_intents_via_LCM(itemsets, min_supp=min_sup)
    keys = mec.list_keys(intents)
    passkeys = mec.list_passkeys(intents)

    children_ordering = ordermod.sort_intents_inclusion(intents)
    parents_ordering = ordermod.inverse_order(children_ordering)

    pseudo_intents = list(dict(ibases.list_pseudo_intents_via_keys(keys.items(), intents)))
    proper_premises = list(ibases.iter_proper_premises_via_keys(intents, keys))

    n_transitive_parents = sum(tparents.count() for tparents in ordermod.close_transitive_subsumption(parents_ordering))
    linearity = indicesmod.linearity_index(n_transitive_parents, len(intents))
    distributivity = indicesmod.distributivity_index(intents, parents_ordering, n_transitive_parents)

    output = dict(
        intents=intents,
        keys=keys, passkeys=passkeys,
        pseudo_intents=pseudo_intents, proper_premises=proper_premises,
        intents_ordering=parents_ordering,
        linearity=linearity, distributivity=distributivity
    )
    if return_itemsets:
        for output_k, output_v in output.items():
            itemset_v = output_v
            if isinstance(output_v, dict):
                itemset_v = dict(zip(bfuncs.bas2isets(output_v.keys()), output_v.values()))
            if isinstance(output_v, list):
                itemset_v = list(bfuncs.bas2isets(output_v))
            output[output_k] = itemset_v

    return output

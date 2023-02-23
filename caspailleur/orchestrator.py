from typing import Dict, Any

import numpy as np

from . import base_functions as bfuncs
from . import implication_bases as ibases
from . import mine_equivalence_classes as mec
from . import order as ordermod
from . import indices as indicesmod


def explore_data(K: np.ndarray, min_sup: float = 1, return_itemsets: bool = True) -> Dict[str, Any]:
    itemsets = bfuncs.np2isets(K)
    n_attrs = K.shape[1]
    attr_extents = [frozenset(bfuncs.ba2iset(ext)) for ext in bfuncs.iter_attribute_extents(K)]

    intents = mec.list_intents_via_LCM(itemsets, min_supp=min_sup, n_attrs=n_attrs)
    intents_ba = [bfuncs.iset2ba(iset, n_attrs) for iset in intents]
    keys = mec.list_keys(intents_ba)
    passkeys = mec.list_passkeys(intents_ba)
    pseudo_intents = ibases.list_pseudo_intents_incremental(attr_extents, intents)

    children_ordering = ordermod.sort_intents_inclusion(intents_ba)
    parents_ordering = ordermod.inverse_order(children_ordering)
    n_transitive_parents = sum(tparents.count() for tparents in ordermod.close_transitive_subsumption(parents_ordering))
    linearity = indicesmod.linearity_index(n_transitive_parents, len(intents))

    proper_premises = list(ibases.iter_proper_premises_via_keys(
        intents, {frozenset(bfuncs.ba2iset(k)): v for k, v in keys.items()}))
    distributivity = indicesmod.distributivity_index(intents_ba, parents_ordering, n_transitive_parents)

    pseudo_intents_ba = [bfuncs.iset2ba(pi, n_attrs) for pi in pseudo_intents]
    proper_premises_ba = [bfuncs.iset2ba(pp, n_attrs) for pp in proper_premises]

    output = dict(
        intents=intents_ba,
        keys=keys, passkeys=passkeys,
        pseudo_intents=pseudo_intents_ba, proper_premises=proper_premises_ba,
        intents_ordering=parents_ordering,
        linearity=linearity, distributivity=distributivity
    )
    if return_itemsets:
        for output_k, output_v in output.items():
            itemset_v = output_v
            if isinstance(output_v, dict):
                itemset_v = {bfuncs.ba2iset(k): v for k, v in output_v.items()}
            if isinstance(output_v, list):
                itemset_v = [bfuncs.ba2iset(v) for v in output_v]
            output[output_k] = itemset_v

    return output

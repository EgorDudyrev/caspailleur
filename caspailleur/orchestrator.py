from typing import Dict, Any, Union

import deprecation
import numpy as np

from . import io
from . import implication_bases as ibases
from . import mine_equivalence_classes as mec
from . import order as ordermod
from . import indices as indicesmod


@deprecation.deprecated(
    deprecated_in="0.2.0", removed_in="0.2.1",
    details="Use functions `mine_implications`, `mine_concepts` and `mine_descriptions` from API module. "
            "They provide easier to work-with output"
)
def explore_data(K: np.ndarray, min_sup: Union[int, float] = 0, return_itemsets: bool = True) -> Dict[str, Any]:
    """One function to output all dependencies in the data

    Parameters
    ----------
    K:
        Binary dataset represented with numpy bool 2D array
    min_sup:
        Minimal support for intents
    return_itemsets:
        A flag whether to return the output as sets of attribute indices or as bitarrays/
        The second option is memory efficient but less interpretable.

    Returns
    -------
    Dictionary with
        intents, keys, passkeys,
        pseudo_intents, proper_premises,
        intents_ordering,
        and linearity, distributivity indices
    """
    itemsets = list(io.np2bas(K))

    intents = mec.list_intents_via_LCM(itemsets, min_supp=min_sup)
    keys = mec.list_keys(intents)
    passkeys = mec.list_passkeys(intents)

    children_ordering = ordermod.sort_intents_inclusion(intents)
    parents_ordering = ordermod.inverse_order(children_ordering)

    pseudo_intents = dict(ibases.list_pseudo_intents_via_keys(keys.items(), intents))
    proper_premises = dict(ibases.iter_proper_premises_via_keys(intents, keys))

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
                itemset_v = dict(zip(io.bas2isets(output_v.keys()), output_v.values()))
            if isinstance(output_v, list):
                itemset_v = list(io.bas2isets(output_v))
            output[output_k] = itemset_v

    return output

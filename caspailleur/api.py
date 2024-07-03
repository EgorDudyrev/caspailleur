"""Module with easy to use general functions for working with Caspailleur"""
from operator import itemgetter
import typing
from typing import Iterator, Iterable, Literal, Union, Optional, get_args
import pandas as pd
from bitarray import frozenbitarray as fbarray

from .base_functions import powerset, extension, intention
from .io import ContextType, to_bitarrays, transpose_context, isets2bas, verbalise, to_absolute_number
from .order import topological_sorting
from . import indices as idxs
from . import definitions
from . import mine_equivalence_classes as mec, implication_bases as ibases


MINE_DESCRIPTION_COLUMN = Literal[
    "description", "extent", "intent",
    "support", "delta_stability",
    "is_closed", "is_key", "is_passkey", "is_proper_premise", "is_pseudo_intent"
]

MINE_CONCEPTS_COLUMN = Literal[
    "extent", "intent", "support", "delta_stability",
    "keys", "passkeys", "proper_premises", "pseudo_intents"
]

MINE_IMPLICATIONS_COLUMN = Literal[
    "premise", "conclusion", "conclusion_full", "extent", "support"
]

BASIS_NAME = Literal[
    "Proper Premise", "Canonical Direct", "Karell",
    "Pseudo-Intent", "Canonical", "Duquenne-Guigues", "Minimum",
]


def _setup_colnames_to_compute(
        all_columns,
        columns_to_compute: Union[list[str], Literal['all'], None],
        dependencies: dict[Union[str, tuple[str, bool]], set[str]], return_all_computed: bool
) -> tuple[set[str], list[str]]:
    columns_to_return = list(get_args(all_columns))
    if columns_to_compute is not None and columns_to_compute != 'all':
        columns_to_return = list(columns_to_compute)
    columns_to_compute = set(columns_to_return)

    assert columns_to_compute <= set(get_args(all_columns)), \
        f"The following elements were asked for but cannot be computed {columns_to_compute - set(all_columns)}. " \
        f"However, only the following columns can be chosen: {all_columns}."

    for premise, conclusion in dependencies.items():
        if not isinstance(premise, str):
            if not premise[1]:
                continue
            premise = premise[0]
        columns_to_compute.update(conclusion if premise in columns_to_compute or premise == '' else set())

    if return_all_computed:
        columns_to_return += sorted(set(columns_to_compute) - set(columns_to_return), key=list(get_args(all_columns)).index)
    return columns_to_compute, columns_to_return


def iter_descriptions(
        data: ContextType,
        to_compute: Optional[Union[list[MINE_DESCRIPTION_COLUMN], Literal['all']]] = (
                "description", "extent", "intent",
                "support", "delta_stability",
                "is_closed", "is_key", "is_passkey", "is_proper_premise",
        )
) -> Iterator[dict]:
    bitarrays, objects, attributes = to_bitarrays(data)
    attr_extents = transpose_context(bitarrays)

    sub_pseudo_intents = []

    def get_vals_for_column(column_name: MINE_DESCRIPTION_COLUMN):
        if column_name == 'description':
            return set(verbalise(description_idxs, attributes))
        if column_name == 'extent':
            return set(verbalise(extent_ba, objects))
        if column_name == 'intent':
            return set(verbalise(definitions.closure(description_idxs, attr_extents), attributes))
        if column_name == 'support':
            return idxs.support_by_description(description_idxs, attr_extents, extent_ba)
        if column_name == 'delta_stability':
            return idxs.delta_stability_by_description(description_idxs, attr_extents)
        if column_name == 'is_closed':
            return definitions.is_closed(description_idxs, attr_extents)
        if column_name == 'is_key':
            return definitions.is_key(description_idxs, attr_extents)
        if column_name == 'is_passkey':
            return definitions.is_passkey(description_idxs, attr_extents)
        if column_name == 'is_proper_premise':
            return definitions.is_proper_premise(description_idxs, attr_extents)
        if column_name == 'is_pseudo_intent':
            return definitions.is_pseudo_intent(description_idxs, attr_extents, sub_pseudo_intents)
        raise NotImplementedError(f"Something's gone wrong in the code, asked for column: {column_name}")

    for description_idxs in powerset(range(len(attributes))):
        extent_ba = fbarray(extension(description_idxs, attr_extents))
        stats = {colname: get_vals_for_column(colname) for colname in to_compute}

        yield stats
        if 'is_pseudo_intent' in to_compute and stats['is_pseudo_intent']:
            sub_pseudo_intents.append(description_idxs)


def mine_descriptions(
        data: ContextType,
        min_support: Union[int, float] = 0,
        to_compute: Optional[Union[list[MINE_DESCRIPTION_COLUMN], Literal['all']]] = (
                "description", "extent", "intent",
                "support", "delta_stability",
                "is_closed", "is_key", "is_passkey", "is_proper_premise",
        ),
        return_every_computed_column: bool = False
) -> pd.DataFrame:
    """Mine all frequent descriptions and their characteristics


    Note: If you want to look at only the most stable descriptions, then use "mine_concepts" function,
     as all stable descriptions are concept intents.
    """
    ####################################################
    # Computing what columns and parameters to compute #
    ####################################################
    col_dependencies: dict[MINE_DESCRIPTION_COLUMN, set[MINE_DESCRIPTION_COLUMN]] = {
        'is_pseudo_intent': {'is_proper_premise', 'intent'},
        'is_proper_premise': {'intent', 'is_key'},
        'is_key': {'intent'},
        'is_passkey': {'intent'},
        'is_closed': {'intent'},
        ('', min_support > 0): {'support'},  # always compute support if min_support > 0
        'support': {'extent'},
        'delta_stability': {'extent'},
        'intent': {'extent'},
    }
    to_compute, cols_to_return = _setup_colnames_to_compute(
        MINE_DESCRIPTION_COLUMN, to_compute, col_dependencies, return_every_computed_column)

    ################################
    # Compute the required columns #
    ################################
    bitarrays, objects, attributes = to_bitarrays(data)
    attr_extents = transpose_context(bitarrays)
    n_objects = len(objects)
    min_support = to_absolute_number(min_support, n_objects)

    descriptions_ba = list(isets2bas(powerset(range(len(attributes))), len(attributes)))
    if 'extent' in to_compute:
        extents_ba = [fbarray(extension(descr, attr_extents)) for descr in descriptions_ba]
    if min_support > 0:
        descriptions_ba, extents_ba = zip(*[(descr, ext) for descr, ext in zip(descriptions_ba, extents_ba)
                                              if ext.count() >= min_support])
    column_description = [verbalise(descr, attributes) for descr in descriptions_ba]
    if 'extent' in to_compute:
        column_extent = [verbalise(extent, objects) for extent in extents_ba]
    if 'intent' in to_compute:
        intents_ba = mec.list_intents_via_LCM(bitarrays, min_supp=min_support)
        ext_int_map = {fbarray(extension(intent, attr_extents)): intent_i for intent_i, intent in enumerate(intents_ba)}
        column_intent = [verbalise(intents_ba[ext_int_map[extent]], attributes) for extent in extents_ba]
    if 'is_key' in to_compute:
        keys_ba = mec.list_keys(intents_ba)
    if 'is_passkey' in to_compute:
        passkeys_ba = mec.list_passkeys(intents_ba)
    if 'is_proper_premise' in to_compute:
        ppremises_ba = dict(ibases.iter_proper_premises_via_keys(intents_ba, keys_ba))
    if 'is_pseudo_intent' in to_compute:
        pintents_ba = dict(ibases.list_pseudo_intents_via_keys(ppremises_ba.items(), intents_ba))
    if 'support' in to_compute:
        column_support = [extent.count() for extent in extents_ba]
    if 'delta_stability' in to_compute:
        column_delta_stability = [idxs.delta_stability_by_description(descr, attr_extents, extent)
                                  for descr, extent in zip(descriptions_ba, extents_ba)]
    if 'is_closed' in to_compute:
        column_is_closed = [intents_ba[ext_int_map[extent]] == descr_ba
                            for descr_ba, extent in zip(descriptions_ba, extents_ba)]
    if 'is_key' in to_compute:
        column_is_key = [descr_ba in keys_ba for descr_ba in descriptions_ba]
    if 'is_passkey' in to_compute:
        column_is_passkey = [descr_ba in passkeys_ba for descr_ba in descriptions_ba]
    if 'is_proper_premise' in to_compute:
        column_is_proper_premise = [descr_ba in ppremises_ba for descr_ba in descriptions_ba]
    if 'is_pseudo_intent' in to_compute:
        column_is_pseudo_intent = [descr_ba in pintents_ba for descr_ba in descriptions_ba]

    locals_ = locals()
    return pd.DataFrame({f: locals_[f"column_{f}"] for f in cols_to_return})


def mine_concepts(
        data: ContextType,
        to_compute: Optional[Union[list[MINE_CONCEPTS_COLUMN], Literal['all']]] = (
                'extent', 'intent', 'support', 'delta_stability', 'keys', 'passkeys', 'proper_premises'),
        min_support: Union[int, float] = 0,
        min_delta_stability: Union[int, float] = 0, n_stable_concepts: Optional[int] = None,
        use_tqdm: bool = False,
        return_every_computed_column: bool = False
) -> pd.DataFrame:
    ##################################################
    # Computing what columns and parameters to compute
    ##################################################
    # whether to compute all concepts whose support is higher min_support
    compute_join_semilattice = min_delta_stability == 0 and n_stable_concepts is None
    col_dependencies: dict[MINE_CONCEPTS_COLUMN, set[MINE_CONCEPTS_COLUMN]] = {
        'pseudo_intents': {'proper_premises', 'intent'},
        'proper_premises': {'intent', 'keys'},
        'keys': {'intent'},
        'passkeys': {'intent'},
        ('keys', not compute_join_semilattice): {'extent'},  # extents are not required when computing join semilattice
        ('passkeys', not compute_join_semilattice): {'extent'},
        'support': {'extent'},
        'delta_stability': {'intent', 'extent'},
        'extent': {'intent'},
    }
    to_compute, cols_to_return = _setup_colnames_to_compute(
        MINE_CONCEPTS_COLUMN, to_compute, col_dependencies, return_every_computed_column)

    #################################
    # Running the (long) computations
    #################################
    bitarrays, objects, attributes = to_bitarrays(data)
    attr_extents = transpose_context(bitarrays)

    def verbalise_descriptions(bas):
        return [verbalise(ba, attributes) for ba in bas]

    def group_by_concept(pairs: Iterable[tuple[fbarray, int]], n_cncpts: int) -> list[list[set[str]]]:
        per_concept = [[] for _ in range(n_cncpts)]
        for ba, cncpt_i in pairs:
            per_concept[cncpt_i].append(ba)
        return per_concept

    if 'intent' in to_compute:
        if compute_join_semilattice:
            intents_ba = mec.list_intents_via_LCM(bitarrays, min_supp=min_support)
        else:
            n_objects = len(objects)
            stable_extents = mec.list_stable_extents_via_gsofia(
                attr_extents, n_objects, min_delta_stability, n_stable_concepts,
                min_supp=to_absolute_number(min_support, n_objects), n_attributes=len(attr_extents)
            )
            intents_ba = [intention(extent, attr_extents) for extent in stable_extents]

        intents_ba = topological_sorting(intents_ba)[0]
        n_concepts = len(intents_ba)
        column_intent = verbalise_descriptions(intents_ba)
    if 'extent' in to_compute:
        extents_ba = [fbarray(extension(intent, attr_extents)) for intent in intents_ba]
        column_extent = [verbalise(extent, objects) for extent in extents_ba]
    if 'keys' in to_compute:
        if compute_join_semilattice:
            keys_ba = mec.list_keys(intents_ba)
        else:
            keys_ba = mec.list_keys_for_extents(extents_ba, attr_extents)
        column_keys = [verbalise_descriptions(keys) for keys in group_by_concept(keys_ba.items(), n_concepts)]
    if 'passkeys' in to_compute:
        if compute_join_semilattice:
            passkeys_ba = mec.list_passkeys(intents_ba)
        else:
            passkeys_ba = mec.list_passkeys_for_extents(extents_ba, attr_extents)
        column_passkeys = [verbalise_descriptions(pkeys) for pkeys in group_by_concept(passkeys_ba.items(), n_concepts)]
    if 'proper_premises' in to_compute:
        ppremises_ba = dict(ibases.iter_proper_premises_via_keys(intents_ba, keys_ba))
        column_proper_premises = [verbalise_descriptions(pps) for pps in group_by_concept(ppremises_ba.items(), n_concepts)]
    if 'pseudo_intents' in to_compute:
        pintents_ba = dict(ibases.list_pseudo_intents_via_keys(
            ppremises_ba.items(), intents_ba, use_tqdm=use_tqdm, n_keys=len(ppremises_ba)))
        column_pseudo_intents = [verbalise_descriptions(pis) for pis in group_by_concept(pintents_ba.items(), n_concepts)]
    if 'support' in to_compute:
        column_support = [extent.count() for extent in extents_ba]
    if 'delta_stability' in to_compute:
        column_delta_stability = [idxs.delta_stability_by_description(descr, attr_extents, extent_ba)
                                  for descr, extent_ba in zip(intents_ba, extents_ba)]

    locals_ = locals()
    return pd.DataFrame({f: locals_[f"column_{f}"] for f in cols_to_return})


def mine_implications(
        data: ContextType, basis_name: BASIS_NAME = 'Proper Premise',
        unit_base: bool = False,
        to_compute: Optional[Union[list[MINE_IMPLICATIONS_COLUMN], Literal['all']]] = 'all',
) -> pd.DataFrame:
    assert basis_name in get_args(BASIS_NAME), \
        f"You asked for '{basis_name}' basis. But only the following bases are supported: {BASIS_NAME}"
    if basis_name in {'Canonical Direct', "Karell"}:
        basis_name = 'Proper Premise'
    if basis_name in {'Canonical', 'Duquenne-Guigues', 'Minimum'}:
        basis_name = 'Pseudo-Intent'

    dependencies = {
        "support": {'extent'},
    }

    to_compute, cols_to_return = _setup_colnames_to_compute(
        MINE_IMPLICATIONS_COLUMN, to_compute, dependencies, return_all_computed=True)

    ############################
    # Compute the (unit) basis #
    ############################
    bitarrays, objects, attributes = to_bitarrays(data)
    attr_extents = transpose_context(bitarrays)

    intents_ba = mec.list_intents_via_LCM(bitarrays)
    keys_ba = mec.list_keys(intents_ba)
    ppremises_ba = list(ibases.iter_proper_premises_via_keys(intents_ba, keys_ba))
    basis = ppremises_ba
    if basis_name == 'Pseudo-Intent':
        basis = ibases.list_pseudo_intents_via_keys(ppremises_ba, intents_ba)

    # compute pseudo-closures to reduce the conclusions by attributes implied by other implications
    if 'conclusion' in to_compute:
        pseudo_closures = [
            ibases.saturate(premise, basis[:impl_i] + basis[impl_i + 1:], intents_ba)
            for impl_i, (premise, intent_i) in enumerate(basis)
        ]
        basis = [(premise, intents_ba[intent_i] & ~pintent, intent_i)
                 for (premise, intent_i), pintent in zip(basis, pseudo_closures)]
    else:
        basis = [(premise, None, intent_i) for premise, intent_i in basis]

    if unit_base and 'conclusion' in to_compute:
        single_attrs = to_bitarrays([{i} for i in range(len(attributes))])[0]
        basis = [(premise, single_attrs[attr_i], intent_i)
                 for premise, conclusion, intent_i in basis for attr_i in conclusion.search(True)]
    premises, conclusions, intents_idxs = zip(*basis)

    ####################################
    # Prepare values for the dataframe #
    ####################################
    def verbalise_descriptions(bas):
        return [verbalise(ba, attributes) for ba in bas]

    if 'premise' in to_compute:
        column_premise = verbalise_descriptions(premises)
    if 'conclusion' in to_compute:
        column_conclusion = verbalise_descriptions(conclusions)
        if unit_base:
            column_conclusion = [list(conclusion)[0] for conclusion in column_conclusion]
    if 'conclusion_full' in to_compute:
        column_conclusion_full = verbalise_descriptions(map(intents_ba.__getitem__, intents_idxs))
    if "extent" in to_compute:
        extents_ba = [fbarray(extension(intent, attr_extents)) for intent in intents_ba]
        int_ext_map = dict(zip(intents_ba, extents_ba))
        column_extent = [verbalise(int_ext_map[intents_ba[intent_i]], objects) for intent_i in intents_idxs]
    if "support" in to_compute:
        column_support = [int_ext_map[intents_ba[intent_i]].count() for intent_i in intents_idxs]

    locals_ = locals()
    return pd.DataFrame({f: locals_[f"column_{f}"] for f in cols_to_return})
